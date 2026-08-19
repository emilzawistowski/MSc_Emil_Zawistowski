# =============================================================================
#
# GOAL
# ----
# This is the PRIMARY, pre-registered test of Hypothesis 4 (sec:h4 /
# sec:results_h4 in the thesis), which explicitly specifies:
#   "This discriminability will be tested using a classification pipeline
#    employing Common Spatial Patterns (CSP) and Linear Discriminant
#    Analysis (LDA). Using leave-one-subject-out validation, it is
#    predicted that the model will achieve a decoding accuracy for
#    Near vs. Far deviants that significantly exceeds the 50% chance
#    level. The robustness of this classification will be validated
#    through a permutation test with alpha = 0.05..."
#
# This differs from 04_decoding.py (SVM, within-subject 5-fold
# StratifiedKFold, ROC-AUC, sliding time window) in three ways that
# matter for what each script can claim:
#
#   1. Feature extraction : CSP spatial filters (this script) vs. raw
#      per-timepoint channel amplitudes (04_decoding.py).
#   2. Cross-validation    : leave-one-subject-out, i.e. the classifier
#      is trained on N-1 participants and tested on the held-out
#      participant (this script) vs. within-subject 5-fold CV, i.e.
#      each participant's own trials are split into folds and averaged
#      into a per-participant score, only pooled across people
#      afterwards for group statistics (04_decoding.py).
#   3. Question answered   : "Is there a distance-specific neural code
#      that GENERALISES ACROSS PARTICIPANTS?" (this script, LOSO) vs.
#      "Is the near/far pattern separable WITHIN each participant's
#      own trials?" (04_decoding.py, within-subject).
#
# Both are legitimate, complementary analyses; this script is the one
# that matches the pre-registered Methods text and should be reported
# as the confirmatory test of H4. 04_decoding.py should be reframed in
# the thesis as a supplementary/exploratory sensitivity analysis.
#
# CSP is fit on a single (or a small number of) time window(s) rather
# than at every sliding time point, because CSP requires a fixed
# spatial covariance estimated over a window of samples (it is not a
# per-timepoint sliding method the way SlidingEstimator is). Two
# variants are computed:
#   (a) "windowed"     : one CSP+LDA model per fixed sub-window,
#                        stepped across the MMN latency window
#                        (100-280 ms), giving a time-resolved LOSO
#                        accuracy curve broadly comparable in spirit
#                        to the SlidingEstimator figure from
#                        04_decoding.py.
#   (b) "single-window" : one CSP+LDA model fit on the full
#                        pre-registered MMN latency window
#                        (100-280 ms) as a single feature block - this
#                        is the primary, single-number confirmatory
#                        test referred to in the Methods text ("a
#                        decoding accuracy ... that significantly
#                        exceeds the 50% chance level").
#
# USAGE
# -----
#   python 04b_decoding_csp_lda.py
#   python 04b_decoding_csp_lda.py --participants P01 P02 P03
#   python 04b_decoding_csp_lda.py --n-permutations 1000
#
# OUTPUT
# ------
#   results/tables/decoding_csp_lda/loso_fold_results.json
#   results/tables/decoding_csp_lda/loso_summary.json
#   results/tables/decoding_csp_lda/windowed_loso_scores.npy
#   results/figures/decoding/csp_lda_loso_timecourse.png
#   results/figures/decoding/csp_patterns_topomap.png
# =============================================================================

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


# =============================================================================
# 0. DECODING-SPECIFIC PARAMETERS
# =============================================================================

CLASS_A_LABEL = 'looming'     # Block A deviant  (S 12, 2 m / near)
CLASS_B_LABEL = 'receding'    # Block B deviant  (S 22, 5 m / far)

# Primary confirmatory window: full pre-registered MMN latency window,
# matching the Methods/Results text (100-280 ms).
DECODE_TMIN = cfg.MMN_LATENCY_TMIN   # 0.100 s
DECODE_TMAX = cfg.MMN_LATENCY_TMAX   # 0.280 s

# Secondary, time-resolved analysis: step a fixed-length window across
# the same range so we still get a time-course figure, just built from
# LOSO CSP+LDA windows rather than a per-sample sliding estimator.
WINDOW_LENGTH = 0.050   # s   - 50 ms sub-windows
WINDOW_STEP   = 0.020   # s   - 20 ms step

N_CSP_COMPONENTS = cfg.N_CSP_COMPONENTS   # 6, from config.py
N_PERMUTATIONS   = 1000                   # overridden by --n-permutations
RANDOM_STATE     = cfg.RANDOM_STATE
CHANCE_LEVEL     = 0.5

MIN_EPOCHS_PER_CLASS = 10   # per participant, same floor as 04_decoding.py

DECODING_TABLE_PATH  = cfg.STATS_PATH / 'decoding_csp_lda'
DECODING_FIGURE_PATH = cfg.FIGURES_DECODING_PATH


# =============================================================================
# 1. HELPERS
# =============================================================================

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
        print(f"      EXCLUDED: {participant_id} block {block} "
              f"(see excluded_recordings.csv) - skipping.")
        return None

    fname = f"{participant_id}_block{block}_{condition}_epo.fif"
    fpath = cfg.PREPROCESSED_PATH / fname
    if not fpath.exists():
        print(f"      WARNING: Missing file {fname} - skipping.")
        return None
    return fpath


def load_looming_receding_epochs(participant_id: str) -> mne.Epochs | None:
    """
    Load and combine Block A deviant ('looming') and Block B deviant
    ('receding') epochs for one participant. Identical logic to
    04_decoding.py's loader, duplicated here so this script has no
    import-time dependency on 04_decoding.py.
    """
    print(f"\n  [Load] {participant_id}: looming (A-dev) vs. receding (B-dev)")

    path_a = _find_epo_file(participant_id, 'A', 'dev')
    path_b = _find_epo_file(participant_id, 'B', 'dev')
    if path_a is None or path_b is None:
        print(f"      SKIP {participant_id}: missing looming and/or receding epochs.")
        return None

    try:
        epo_a = mne.read_epochs(path_a, preload=True, verbose=False)
        epo_b = mne.read_epochs(path_b, preload=True, verbose=False)
    except Exception as e:
        print(f"      ERROR reading epochs for {participant_id}: {e}")
        return None

    if len(epo_a) < MIN_EPOCHS_PER_CLASS or len(epo_b) < MIN_EPOCHS_PER_CLASS:
        print(f"      SKIP {participant_id}: too few trials "
              f"(looming={len(epo_a)}, receding={len(epo_b)}, "
              f"minimum={MIN_EPOCHS_PER_CLASS}).")
        return None

    common_ch = [ch for ch in epo_a.ch_names if ch in epo_b.ch_names]
    if len(common_ch) < len(epo_a.ch_names) or len(common_ch) < len(epo_b.ch_names):
        print(f"      WARNING: Channel mismatch; restricting to "
              f"{len(common_ch)} common channels.")
    epo_a = epo_a.copy().pick(common_ch)
    epo_b = epo_b.copy().pick(common_ch)

    epo_a.events[:, 2] = 1
    epo_a.event_id = {CLASS_A_LABEL: 1}
    epo_b.events[:, 2] = 2
    epo_b.event_id = {CLASS_B_LABEL: 2}

    epochs = mne.concatenate_epochs([epo_a, epo_b], add_offset=True, verbose=False)
    print(f"      Combined: {len(epo_a)} looming + {len(epo_b)} receding "
          f"= {len(epochs)} epochs, {len(common_ch)} channels")

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
            print(f"  UNEXPECTED ERROR loading {pid}: {e}")
            traceback.print_exc()

    if len(all_epochs) < 3:
        return all_epochs

    # Intersect channel sets across ALL participants (CSP/LDA weights
    # learned on N-1 participants must apply unchanged to the held-out
    # participant, so the feature space -- the channel list and order --
    # must be identical for everyone).
    common_ch = set(next(iter(all_epochs.values())).ch_names)
    for epo in all_epochs.values():
        common_ch &= set(epo.ch_names)
    common_ch = [ch for ch in next(iter(all_epochs.values())).ch_names
                 if ch in common_ch]

    print(f"\n  Common channels across all {len(all_epochs)} participants: "
          f"{len(common_ch)} -> {common_ch}")

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


# =============================================================================
# 2. LOSO CSP+LDA CORE
# =============================================================================

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

        # Guard: need both classes present in the training fold (should
        # always hold with >=2 remaining participants, but check anyway).
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            print(f"      Fold {fold_i}: skipped (missing a class in "
                  f"train or test split).")
            continue

        csp = CSP(n_components=n_csp, reg='ledoit_wolf', log=True,
                  norm_trace=False)
        lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        clf = make_pipeline(csp, lda)

        try:
            clf.fit(X_train, y_train)
        except Exception as e:
            print(f"      Fold {fold_i}: CSP/LDA fit failed ({e}); skipping.")
            continue

        y_pred = clf.predict(X_test)
        # decision_function gives a continuous score suitable for AUC;
        # LDA's classes_ ordering determines the sign convention.
        y_score = clf.decision_function(X_test)

        acc = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_score)
        except ValueError:
            auc = np.nan   # can happen if y_test is single-class (shouldn't, given guard above)

        fold_results.append({
            'held_out_index': held_out_group,
            'n_test_trials': int(len(y_test)),
            'accuracy': float(acc),
            'auc': float(auc) if not np.isnan(auc) else None,
        })

        y_true_pooled.append(y_test)
        y_pred_pooled.append(y_pred)
        y_score_pooled.append(y_score)

        print(f"      Fold {fold_i} (held-out group {held_out_group}): "
              f"n={len(y_test)}, acc={acc:.3f}, "
              f"auc={'nan' if np.isnan(auc) else f'{auc:.3f}'}")

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


# =============================================================================
# 3. PERMUTATION TEST (label-shuffling within participant, LOSO refit)
# =============================================================================

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
            print(f"      Permutation {p + 1}/{n_permutations} done "
                  f"(running mean null acc = {np.nanmean(perm_accuracies[:p+1]):.3f})")

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


# =============================================================================
# 4. TIME-RESOLVED (WINDOWED) LOSO ANALYSIS
# =============================================================================

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
        print(f"\n  [Windowed LOSO] {w_start*1000:.0f}-{w_end*1000:.0f} ms")
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


# =============================================================================
# 5. VISUALISATION
# =============================================================================

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
    print(f"\n  Figure saved: {out_path}")


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
    pos = np.array([ch['loc'][:3] for ch in info['chs'] if ch['kind'] == 2])  # FIFFV_EEG_CH
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
        # Same fix as 02_erp_analysis.py topomaps: default sphere='auto'
        # undersizes the head outline for this 32-ch montage (FT9/FT10/
        # TP9/TP10 sit outside it). See _sphere_covering_all_channels().
        sphere = _sphere_covering_all_channels(info)
        fig = csp.plot_patterns(info, ch_type='eeg', units='AU', size=1.5,
                                  sphere=sphere, show=False)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"  CSP patterns figure saved: {out_path}")
    except Exception as e:
        print(f"  WARNING: could not plot CSP patterns ({e}); skipping "
              f"(this is a visualisation aid only, not required for the "
              f"confirmatory statistic).")


# =============================================================================
# 6. MAIN PIPELINE
# =============================================================================

def run_decoding(participants: list[str], n_permutations: int, n_jobs: int = 1) -> None:
    DECODING_TABLE_PATH.mkdir(parents=True, exist_ok=True)
    DECODING_FIGURE_PATH.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  PRE-REGISTERED H4 TEST: CSP + LDA, leave-one-subject-out")
    print("  looming (Block A dev) vs. receding (Block B dev)")
    print(f"  Confirmatory window: {DECODE_TMIN*1000:.0f}-{DECODE_TMAX*1000:.0f} ms "
          f"| CSP components: {N_CSP_COMPONENTS} | permutations: {n_permutations}")
    print("=" * 70)

    all_epochs = load_all_participants(participants)
    n_loaded = len(all_epochs)
    n_skipped = len(participants) - n_loaded
    print(f"\n  Participants loaded: {n_loaded} | skipped: {n_skipped} "
          f"(out of {len(participants)})")

    if n_loaded < 3:
        print("  ERROR: Fewer than 3 participants available - LOSO requires "
              "at least 3 (2 to train on, 1 held out). Aborting.")
        return

    # --- Primary confirmatory analysis: single full MMN-latency window ---
    print("\n" + "-" * 70)
    print("  PRIMARY ANALYSIS: single window, 100-280 ms (pre-registered)")
    print("-" * 70)
    X, y, groups, pid_list = build_group_arrays(all_epochs, DECODE_TMIN, DECODE_TMAX)
    print(f"  Total trials: {X.shape[0]} | channels: {X.shape[1]} | "
          f"time samples/window: {X.shape[2]} | participants (groups): {len(pid_list)}")

    primary_result = run_loso_csp_lda(X, y, groups, n_csp=N_CSP_COMPONENTS)

    if primary_result['pooled_accuracy'] is None:
        print("  ERROR: LOSO CSP+LDA produced no valid folds. Aborting.")
        return

    print(f"\n  Pooled LOSO accuracy = {primary_result['pooled_accuracy']:.3f} "
          f"(per-fold mean = {primary_result['per_fold_mean_accuracy']:.3f} "
          f"+/- {primary_result['per_fold_sd_accuracy']:.3f})")
    if primary_result['pooled_auc'] is not None:
        print(f"  Pooled LOSO AUC      = {primary_result['pooled_auc']:.3f}")

    # --- Permutation test on the primary statistic ---
    print("\n" + "-" * 70)
    print(f"  PERMUTATION TEST ({n_permutations} permutations, alpha = 0.05)")
    print("-" * 70)
    perm_result = permutation_test_loso(
        X, y, groups,
        observed_accuracy=primary_result['pooled_accuracy'],
        n_permutations=n_permutations,
        n_csp=N_CSP_COMPONENTS,
    )
    print(f"\n  Null distribution: mean = {perm_result['null_mean']:.3f}, "
          f"sd = {perm_result['null_sd']:.3f}")
    print(f"  Observed accuracy = {perm_result['observed_accuracy']:.3f}, "
          f"p = {perm_result['p_value']:.4f} "
          f"({'SIGNIFICANT' if perm_result['significant_at_05'] else 'not significant'} at alpha=0.05)")

    # --- Secondary, time-resolved LOSO analysis (descriptive figure only) ---
    print("\n" + "-" * 70)
    print("  SECONDARY ANALYSIS: time-resolved (windowed) LOSO accuracy")
    print("-" * 70)
    windowed = run_windowed_loso(all_epochs, n_csp=N_CSP_COMPONENTS)

    # --- Save everything ---
    fold_results_path = DECODING_TABLE_PATH / 'loso_fold_results.json'
    with open(fold_results_path, 'w', encoding='utf-8') as f:
        json.dump(_to_python(primary_result['fold_results']), f, indent=2, ensure_ascii=False)
    print(f"\n  Per-fold results saved: {fold_results_path}")

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
    print(f"  Summary saved: {summary_path}")

    # --- Figures ---
    fig_path = DECODING_FIGURE_PATH / 'csp_lda_loso_timecourse.png'
    plot_windowed_loso(windowed, primary_result, fig_path)

    patterns_path = DECODING_FIGURE_PATH / 'csp_patterns_topomap.png'
    plot_csp_patterns(all_epochs, N_CSP_COMPONENTS, patterns_path)

    print("\n" + "=" * 70)
    print("  DONE.")
    print(f"  H4 (pre-registered CSP+LDA/LOSO): pooled accuracy = "
          f"{primary_result['pooled_accuracy']:.3f}, permutation p = "
          f"{perm_result['p_value']:.4f} -> "
          f"{'SUPPORTED' if perm_result['significant_at_05'] and primary_result['pooled_accuracy'] > CHANCE_LEVEL else 'NOT SUPPORTED'}")
    print("=" * 70)


# =============================================================================
# ENTRY POINT
# =============================================================================

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