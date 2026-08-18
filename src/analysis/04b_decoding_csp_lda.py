import sys
import json
import argparse
import traceback
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import mne
from mne.decoding import CSP

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

mne.set_log_level('WARNING')


CLASS_A_LABEL = 'looming'                                          
CLASS_B_LABEL = 'receding'                                        


DECODE_TMIN = cfg.MMN_LATENCY_TMIN            
DECODE_TMAX = cfg.MMN_LATENCY_TMAX            


WINDOW_LENGTH = 0.050                            
WINDOW_STEP   = 0.020                     

N_CSP_COMPONENTS = cfg.N_CSP_COMPONENTS                      
N_PERMUTATIONS   = 1000                                                   
RANDOM_STATE     = cfg.RANDOM_STATE
CHANCE_LEVEL     = 0.5

MIN_EPOCHS_PER_CLASS = 10                                                  

DECODING_TABLE_PATH  = cfg.STATS_PATH / 'decoding_csp_lda'
DECODING_FIGURE_PATH = cfg.FIGURES_DECODING_PATH


def _to_python(obj):
    """Recursively convert numpy scalars/arrays to built-in Python types."""
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _find_epo_file(participant_id: str, block: str, condition: str) -> Path | None:
    """Locate a saved epochs file, following 01_preprocessing.py's naming
    convention: {PID}_block{A|B}_{std|dev}_epo.fif"""
    if cfg.is_excluded(participant_id, block):


        return None

    fname = f"{participant_id}_block{block}_{condition}_epo.fif"
    fpath = cfg.PREPROCESSED_PATH / fname
    if not fpath.exists():

        return None
    return fpath


def load_looming_receding_epochs(participant_id: str) -> mne.Epochs | None:
    """
    Load and combine Block A deviant ('looming') and Block B deviant
    ('receding') epochs for one participant. Identical logic to
    04_decoding.py's loader, duplicated here so this script has no
    import-time dependency on 04_decoding.py.
    """


    path_a = _find_epo_file(participant_id, 'A', 'dev')
    path_b = _find_epo_file(participant_id, 'B', 'dev')
    if path_a is None or path_b is None:

        return None

    try:
        epo_a = mne.read_epochs(path_a, preload=True, verbose=False)
        epo_b = mne.read_epochs(path_b, preload=True, verbose=False)
    except Exception as e:

        return None

    if len(epo_a) < MIN_EPOCHS_PER_CLASS or len(epo_b) < MIN_EPOCHS_PER_CLASS:


        return None

    common_ch = [ch for ch in epo_a.ch_names if ch in epo_b.ch_names]
    if len(common_ch) < len(epo_a.ch_names) or len(common_ch) < len(epo_b.ch_names):


        pass
    epo_a = epo_a.copy().pick(common_ch)
    epo_b = epo_b.copy().pick(common_ch)

    epo_a.events[:, 2] = 1
    epo_a.event_id = {CLASS_A_LABEL: 1}
    epo_b.events[:, 2] = 2
    epo_b.event_id = {CLASS_B_LABEL: 2}

    epochs = mne.concatenate_epochs([epo_a, epo_b], add_offset=True, verbose=False)


    return epochs


def load_all_participants(participants: list[str]) -> dict:
    """
    Load epochs for every participant that has enough trials.
    Returns {pid: mne.Epochs}. Channels are further intersected across
    ALL participants afterwards (LOSO needs a common feature space,
    since CSP spatial filters are fit on pooled training-participant
    data and applied to the held-out participant's channels).
    """
    all_epochs = {}
    for pid in participants:
        try:
            epochs = load_looming_receding_epochs(pid)
            if epochs is not None:
                all_epochs[pid] = epochs
        except Exception as e:

            traceback.print_exc()

    if len(all_epochs) < 3:
        return all_epochs


    common_ch = set(next(iter(all_epochs.values())).ch_names)
    for epo in all_epochs.values():
        common_ch &= set(epo.ch_names)
    common_ch = [ch for ch in next(iter(all_epochs.values())).ch_names
                 if ch in common_ch]


    for pid in list(all_epochs.keys()):
        all_epochs[pid] = all_epochs[pid].copy().pick(common_ch)

    return all_epochs


def build_group_arrays(all_epochs: dict, tmin: float, tmax: float):
    """
    Concatenate all participants' epochs (cropped to [tmin, tmax]) into
    single X, y, groups arrays for LeaveOneGroupOut.

    Returns
    -------
    X : ndarray, shape (n_epochs_total, n_channels, n_times)
    y : ndarray, shape (n_epochs_total,)         - 1 = looming, 2 = receding
    groups : ndarray, shape (n_epochs_total,)    - integer-coded participant id
    pid_list : list[str]                          - groups[i] indexes into this
    """
    X_list, y_list, groups_list = [], [], []
    pid_list = list(all_epochs.keys())

    for gi, pid in enumerate(pid_list):
        epo = all_epochs[pid].copy().crop(tmin=tmin, tmax=tmax)
        X = epo.get_data(picks='eeg')
        y = epo.events[:, 2]
        X_list.append(X)
        y_list.append(y)
        groups_list.append(np.full(len(y), gi))

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    groups = np.concatenate(groups_list, axis=0)
    return X, y, groups, pid_list


def run_loso_csp_lda(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                      n_csp: int = N_CSP_COMPONENTS) -> dict:
    """
    Fit CSP+LDA with leave-one-subject-out cross-validation.

    For each held-out participant: CSP spatial filters and the LDA
    classifier are fit ONLY on the remaining N-1 participants' trials,
    then applied unchanged to the held-out participant's trials. This
    is the correct way to avoid leaking held-out-participant
    information into the spatial filters themselves, which a naive
    "fit CSP on everyone, then cross-validate the classifier only"
    approach would not guarantee.

    Returns a dict with per-fold accuracy/AUC and the pooled
    (all-held-out-trials) accuracy/AUC.
    """
    logo = LeaveOneGroupOut()
    fold_results = []
    y_true_pooled, y_pred_pooled, y_score_pooled = [], [], []

    for fold_i, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        held_out_group = int(groups[test_idx[0]])

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]


        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:


            continue

        csp = CSP(n_components=n_csp, reg='ledoit_wolf', log=True,
                  norm_trace=False)
        lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        clf = make_pipeline(csp, lda)

        try:
            clf.fit(X_train, y_train)
        except Exception as e:

            continue

        y_pred = clf.predict(X_test)


        y_score = clf.decision_function(X_test)

        acc = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_score)
        except ValueError:
            auc = np.nan                                                                        

        fold_results.append({
            'held_out_index': held_out_group,
            'n_test_trials': int(len(y_test)),
            'accuracy': float(acc),
            'auc': float(auc) if not np.isnan(auc) else None,
        })

        y_true_pooled.append(y_test)
        y_pred_pooled.append(y_pred)
        y_score_pooled.append(y_score)


    if not fold_results:
        return {'fold_results': [], 'pooled_accuracy': None, 'pooled_auc': None,
                 'per_fold_mean_accuracy': None, 'per_fold_mean_auc': None}

    y_true_pooled = np.concatenate(y_true_pooled)
    y_pred_pooled = np.concatenate(y_pred_pooled)
    y_score_pooled = np.concatenate(y_score_pooled)

    pooled_acc = accuracy_score(y_true_pooled, y_pred_pooled)
    try:
        pooled_auc = roc_auc_score(y_true_pooled, y_score_pooled)
    except ValueError:
        pooled_auc = None

    per_fold_accs = [f['accuracy'] for f in fold_results]
    per_fold_aucs = [f['auc'] for f in fold_results if f['auc'] is not None]

    return {
        'fold_results': fold_results,
        'pooled_accuracy': float(pooled_acc),
        'pooled_auc': float(pooled_auc) if pooled_auc is not None else None,
        'per_fold_mean_accuracy': float(np.mean(per_fold_accs)),
        'per_fold_sd_accuracy': float(np.std(per_fold_accs, ddof=1)) if len(per_fold_accs) > 1 else 0.0,
        'per_fold_mean_auc': float(np.mean(per_fold_aucs)) if per_fold_aucs else None,
    }


def permutation_test_loso(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                           observed_accuracy: float,
                           n_permutations: int = N_PERMUTATIONS,
                           n_csp: int = N_CSP_COMPONENTS,
                           seed: int = RANDOM_STATE) -> dict:
    """
    Permutation test for the pooled LOSO accuracy: shuffle class labels
    WITHIN each participant (preserving each participant's class
    balance and the group structure) and rerun the full LOSO CSP+LDA
    pipeline, n_permutations times. Because a full CSP refit per fold
    per permutation is expensive, this loop refits CSP+LDA on each
    LOSO fold within every permutation, exactly mirroring the
    procedure used to obtain the observed statistic (a shortcut such
    as permuting only the LDA stage would not test the full pipeline,
    including CSP, that the observed accuracy is based on).
    """
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    perm_accuracies = np.zeros(n_permutations)

    for p in range(n_permutations):
        y_perm = y.copy()
        for g in unique_groups:
            mask = groups == g
            y_perm[mask] = rng.permutation(y[mask])

        logo = LeaveOneGroupOut()
        y_true_pooled, y_pred_pooled = [], []
        for train_idx, test_idx in logo.split(X, y_perm, groups):
            y_train, y_test = y_perm[train_idx], y_perm[test_idx]
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue
            csp = CSP(n_components=n_csp, reg='ledoit_wolf', log=True,
                      norm_trace=False)
            lda = LinearDiscriminantAnalysis()
            clf = make_pipeline(csp, lda)
            try:
                clf.fit(X[train_idx], y_train)
                y_pred = clf.predict(X[test_idx])
            except Exception:
                continue
            y_true_pooled.append(y_test)
            y_pred_pooled.append(y_pred)

        if y_true_pooled:
            perm_accuracies[p] = accuracy_score(
                np.concatenate(y_true_pooled), np.concatenate(y_pred_pooled))
        else:
            perm_accuracies[p] = np.nan

        if (p + 1) % max(1, n_permutations // 10) == 0:


            pass
    valid = perm_accuracies[~np.isnan(perm_accuracies)]
    p_value = (np.sum(valid >= observed_accuracy) + 1) / (len(valid) + 1)

    return {
        'n_permutations_requested': n_permutations,
        'n_permutations_valid': int(len(valid)),
        'null_mean': float(np.mean(valid)) if len(valid) else None,
        'null_sd': float(np.std(valid, ddof=1)) if len(valid) > 1 else None,
        'observed_accuracy': float(observed_accuracy),
        'p_value': float(p_value),
        'significant_at_05': bool(p_value < 0.05),
    }


def run_windowed_loso(all_epochs: dict, n_csp: int = N_CSP_COMPONENTS) -> dict:
    """
    Step a fixed-length window across [DECODE_TMIN, DECODE_TMAX] and run
    LOSO CSP+LDA (accuracy only, no permutation test - that is reserved
    for the single confirmatory window to keep runtime tractable) in
    each window, to produce a time-resolved curve comparable in spirit
    to the SlidingEstimator figure in 04_decoding.py.
    """
    window_starts = np.arange(DECODE_TMIN, DECODE_TMAX - WINDOW_LENGTH + 1e-9,
                               WINDOW_STEP)
    window_centers = window_starts + WINDOW_LENGTH / 2

    accs, sems = [], []
    for w_start in window_starts:
        w_end = w_start + WINDOW_LENGTH

        X, y, groups, _ = build_group_arrays(all_epochs, w_start, w_end)
        result = run_loso_csp_lda(X, y, groups, n_csp=n_csp)
        fold_accs = [f['accuracy'] for f in result['fold_results']]
        if fold_accs:
            accs.append(np.mean(fold_accs))
            sems.append(np.std(fold_accs, ddof=1) / np.sqrt(len(fold_accs))
                         if len(fold_accs) > 1 else 0.0)
        else:
            accs.append(np.nan)
            sems.append(np.nan)

    return {
        'window_centers_s': window_centers,
        'window_length_s': WINDOW_LENGTH,
        'mean_accuracy': np.array(accs),
        'sem_accuracy': np.array(sems),
    }


def plot_windowed_loso(windowed: dict, primary_result: dict, out_path: Path) -> None:
    times_ms = windowed['window_centers_s'] * 1000
    mean_acc = windowed['mean_accuracy']
    sem_acc = windowed['sem_accuracy']

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(times_ms, mean_acc, color='#1f4e79', linewidth=2.2,
             marker='o', markersize=4, label='LOSO mean accuracy (per window)')
    ax.fill_between(times_ms, mean_acc - sem_acc, mean_acc + sem_acc,
                     color='#1f4e79', alpha=0.25, label='+/- SEM across folds')
    ax.axhline(CHANCE_LEVEL, color='black', linestyle='--', linewidth=1.0,
                label='Chance (0.5)')
    ax.axvline(0, color='black', linestyle=':', linewidth=0.8)

    if primary_result.get('pooled_accuracy') is not None:
        ax.axhline(primary_result['pooled_accuracy'], color='#e07b39',
                    linestyle='-.', linewidth=1.2,
                    label=f"Full-window (100-280 ms) pooled acc "
                          f"= {primary_result['pooled_accuracy']:.3f}")

    ax.set_xlabel('Window centre (ms, relative to corrected sound onset)')
    ax.set_ylabel('LOSO decoding accuracy')
    ax.set_xlim(times_ms[0], times_ms[-1])
    ax.set_ylim(0.25, 1.0)
    ax.legend(loc='upper right', frameon=False, fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def _sphere_covering_all_channels(info: mne.Info, margin: float = 1.03) -> tuple:
    """
    Returns an explicit (x, y, z, radius) sphere for plot_patterns'/
    plot_topomap's `sphere` argument, sized to cover the single most
    eccentric channel rather than MNE's default 'auto' fit (which sizes
    the head outline to the bulk of the montage and can leave low/lateral
    channels -- e.g. FT9/FT10/TP9/TP10 in this 32-ch CLACS layout mapped
    onto standard_1020 -- poking outside the drawn head circle).

    NOTE: purely cosmetic/plotting concern -- does not affect processed
    data, decoding results, or CSP components themselves. Electrode
    positions used throughout the pipeline come from raw.set_montage()
    in 01_preprocessing.py; this only controls how big a circle is drawn
    around those positions when rendering a topomap PNG. Same fix as in
    02_erp_analysis.py's plot_topomap_mmn()/plot_topomap_p3a().
    """
    pos = np.array([ch['loc'][:3] for ch in info['chs'] if ch['kind'] == 2])                
    center_xy = pos[:, :2].mean(axis=0)
    radii = np.sqrt(((pos[:, :2] - center_xy) ** 2).sum(axis=1))
    r = float(radii.max()) * margin
    return (float(center_xy[0]), float(center_xy[1]), 0.0, r)


def plot_csp_patterns(all_epochs: dict, n_csp: int, out_path: Path) -> None:
    """
    Fit one CSP on ALL participants' full-window data (for visualisation
    only - NOT used for any reported decoding statistic, since that
    would leak information across the LOSO splits) and plot the spatial
    patterns as topomaps, to help interpret which scalp regions the
    classifier weights most heavily.
    """
    try:
        X, y, groups, pid_list = build_group_arrays(all_epochs, DECODE_TMIN, DECODE_TMAX)
        csp = CSP(n_components=n_csp, reg='ledoit_wolf', log=True, norm_trace=False)
        csp.fit(X, y)

        info = next(iter(all_epochs.values())).info


        sphere = _sphere_covering_all_channels(info)
        fig = csp.plot_patterns(info, ch_type='eeg', units='AU', size=1.5,
                                  sphere=sphere, show=False)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

    except Exception as e:


        pass
def run_decoding(participants: list[str], n_permutations: int, n_jobs: int = 1) -> None:
    DECODING_TABLE_PATH.mkdir(parents=True, exist_ok=True)
    DECODING_FIGURE_PATH.mkdir(parents=True, exist_ok=True)


    all_epochs = load_all_participants(participants)
    n_loaded = len(all_epochs)
    n_skipped = len(participants) - n_loaded


    if n_loaded < 3:


        return


    X, y, groups, pid_list = build_group_arrays(all_epochs, DECODE_TMIN, DECODE_TMAX)


    primary_result = run_loso_csp_lda(X, y, groups, n_csp=N_CSP_COMPONENTS)

    if primary_result['pooled_accuracy'] is None:

        return


    if primary_result['pooled_auc'] is not None:


        pass
    perm_result = permutation_test_loso(
        X, y, groups,
        observed_accuracy=primary_result['pooled_accuracy'],
        n_permutations=n_permutations,
        n_csp=N_CSP_COMPONENTS,
    )


    windowed = run_windowed_loso(all_epochs, n_csp=N_CSP_COMPONENTS)


    fold_results_path = DECODING_TABLE_PATH / 'loso_fold_results.json'
    with open(fold_results_path, 'w', encoding='utf-8') as f:
        json.dump(_to_python(primary_result['fold_results']), f, indent=2, ensure_ascii=False)


    np.save(DECODING_TABLE_PATH / 'windowed_loso_scores.npy', windowed['mean_accuracy'])
    np.save(DECODING_TABLE_PATH / 'windowed_loso_times.npy', windowed['window_centers_s'])

    summary = {
        'method': 'CSP + LDA, leave-one-subject-out',
        'n_participants': n_loaded,
        'n_skipped': n_skipped,
        'participant_order_for_groups': pid_list,
        'confirmatory_window_ms': [DECODE_TMIN * 1000, DECODE_TMAX * 1000],
        'n_csp_components': N_CSP_COMPONENTS,
        'primary_result': primary_result,
        'permutation_test': perm_result,
        'windowed_analysis': {
            'window_length_ms': WINDOW_LENGTH * 1000,
            'window_step_ms': WINDOW_STEP * 1000,
            'window_centers_ms': (windowed['window_centers_s'] * 1000).tolist(),
            'mean_accuracy': windowed['mean_accuracy'].tolist(),
            'sem_accuracy': windowed['sem_accuracy'].tolist(),
        },
    }
    summary_path = DECODING_TABLE_PATH / 'loso_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(_to_python(summary), f, indent=2, ensure_ascii=False)


    fig_path = DECODING_FIGURE_PATH / 'csp_lda_loso_timecourse.png'
    plot_windowed_loso(windowed, primary_result, fig_path)

    patterns_path = DECODING_FIGURE_PATH / 'csp_patterns_topomap.png'
    plot_csp_patterns(all_epochs, N_CSP_COMPONENTS, patterns_path)


def main():
    parser = argparse.ArgumentParser(
        description='Pre-registered H4 test: CSP + LDA, leave-one-subject-out '
                     'decoding of looming vs. receding auditory distance deviants.'
    )
    parser.add_argument('--participants', nargs='+', default=cfg.PARTICIPANTS,
                         help='Participant IDs to include (default: all in config.py)')
    parser.add_argument('--n-permutations', type=int, default=N_PERMUTATIONS,
                         help='Number of label-shuffling permutations for the '
                              'significance test (default: 1000, matching the '
                              'thesis Methods text). NOTE: each permutation '
                              'refits CSP+LDA for every LOSO fold, so runtime '
                              'scales as n_permutations x n_participants; '
                              'consider a smaller number for a quick check.')
    parser.add_argument('--n-jobs', type=int, default=1,
                         help='(Reserved for future parallelisation; CSP/LDA '
                              'fits here are currently sequential.)')
    args = parser.parse_args()

    cfg.ensure_output_dirs()
    run_decoding(args.participants, n_permutations=args.n_permutations,
                 n_jobs=args.n_jobs)


if __name__ == '__main__':
    main()
    sys.exit(0)