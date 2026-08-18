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
from mne.decoding import SlidingEstimator, cross_val_multiscore, Vectorizer

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from scipy import stats


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

mne.set_log_level('WARNING')


CLASS_A_LABEL = 'looming'                                          
CLASS_B_LABEL = 'receding'                                        


DECODE_TMIN = cfg.MMN_LATENCY_TMIN  
DECODE_TMAX = cfg.MMN_LATENCY_TMAX

N_SPLITS      = 5                           
SCORING       = 'roc_auc'
RANDOM_STATE  = cfg.RANDOM_STATE
N_JOBS        = 1                                  
CHANCE_LEVEL  = 0.5


MIN_EPOCHS_PER_CLASS = 10


DECODING_TABLE_PATH  = cfg.STATS_PATH / 'decoding'
DECODING_FIGURE_PATH = cfg.FIGURES_DECODING_PATH


def _to_python(obj):
    """Recursively convert numpy scalars/arrays to built-in Python types
    (JSON serialisable) - mirrors the helper in 01_preprocessing.py."""
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
    """
    Locate a saved epochs file for one participant/block/condition, following
    the naming convention used by 01_preprocessing.py:
        {PID}_block{A|B}_{std|dev}_epo.fif
    Returns None (with a warning) if the file is missing, rather than raising,
    so a single missing participant/block does not halt the group loop.
    """
    if cfg.is_excluded(participant_id, block):


        return None

    fname = f"{participant_id}_block{block}_{condition}_epo.fif"
    fpath = cfg.PREPROCESSED_PATH / fname
    if not fpath.exists():

        return None
    return fpath


def load_looming_receding_epochs(participant_id: str) -> mne.Epochs | None:
    """
    Load and combine the two deviant conditions needed for this decoding
    contrast:
        Block A deviant (S 12)  -> 'looming'  (near, 2 m)
        Block B deviant (S 22)  -> 'receding' (far, 5 m)

    Returns a single concatenated mne.Epochs object with an events column
    (code 1 = looming, code 2 = receding) usable directly as classifier
    labels, or None if either condition is missing / too few trials.
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


def decode_participant(epochs: mne.Epochs,
                        n_jobs: int = N_JOBS) -> dict:
    """
    Run SlidingEstimator temporal decoding (linear SVM, ROC-AUC,
    5-fold StratifiedKFold) for one participant.

    Returns a dict with:
        'scores'    : ndarray, shape (n_splits, n_times) - per-fold AUC
        'mean'      : ndarray, shape (n_times,)          - fold-averaged AUC
        'times'     : ndarray, shape (n_times,)          - epoch time vector (s)
        'n_looming' : int
        'n_receding': int
    or None on failure.
    """
    try:


        if DECODE_TMIN is not None and DECODE_TMAX is not None:
            epochs_dec = epochs.copy().crop(tmin=DECODE_TMIN, tmax=DECODE_TMAX)
        else:
            epochs_dec = epochs.copy()

        X = epochs_dec.get_data(picks='eeg')                                    
        y = epochs_dec.events[:, 2]                                       
        times = epochs_dec.times

        if X.shape[0] < 2 * MIN_EPOCHS_PER_CLASS:

            return None


        clf = make_pipeline(
            StandardScaler(),
            SVC(kernel='linear', C=1.0, probability=False,
                random_state=RANDOM_STATE),
        )

        time_decoder = SlidingEstimator(
            clf, scoring=SCORING, n_jobs=n_jobs, verbose=False,
        )

        cv = StratifiedKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE,
        )

        scores = cross_val_multiscore(
            time_decoder, X, y, cv=cv, n_jobs=n_jobs,
        )                              

        mean_scores = scores.mean(axis=0)

        n_looming  = int(np.sum(y == 1))
        n_receding = int(np.sum(y == 2))


        return {
            'scores': scores,
            'mean': mean_scores,
            'times': times,
            'n_looming': n_looming,
            'n_receding': n_receding,
        }

    except Exception as e:

        traceback.print_exc()
        return None


def group_level_stats(all_mean_scores: np.ndarray, times: np.ndarray) -> dict:
    """
    Simple, transparent group-level inference:
    one-sample t-test of AUC against chance (0.5) at each time point,
    with an uncorrected p < .05 mask AND a cluster-based permutation test
    (mne.stats.permutation_cluster_1samp_test) for a more defensible,
    multiple-comparisons-corrected result suitable for publication.

    all_mean_scores: ndarray, shape (n_participants, n_times)
    """
    n_participants = all_mean_scores.shape[0]
    mean_auc = all_mean_scores.mean(axis=0)
    sem_auc  = all_mean_scores.std(axis=0, ddof=1) / np.sqrt(n_participants)


    t_vals, p_vals = stats.ttest_1samp(all_mean_scores, popmean=CHANCE_LEVEL, axis=0)
    sig_uncorrected = p_vals < 0.05


    cluster_sig_mask = np.zeros_like(mean_auc, dtype=bool)
    cluster_summary = []
    try:
        from mne.stats import permutation_cluster_1samp_test
        X_centered = all_mean_scores - CHANCE_LEVEL
        t_obs, clusters, cluster_pv, _ = permutation_cluster_1samp_test(
            X_centered, n_permutations=cfg.N_PERMUTATIONS, tail=1,
            seed=RANDOM_STATE, out_type='mask', verbose=False,
        )
        for c_mask, c_p in zip(clusters, cluster_pv):
            if c_p < 0.05:
                cluster_sig_mask |= c_mask
            cluster_summary.append({
                'p_value': float(c_p),
                'significant': bool(c_p < 0.05),
                'time_range_ms': [
                    float(times[c_mask][0] * 1000),
                    float(times[c_mask][-1] * 1000),
                ] if np.any(c_mask) else None,
            })
    except Exception as e:


        pass
    return {
        'mean_auc': mean_auc,
        'sem_auc': sem_auc,
        't_vals': t_vals,
        'p_vals': p_vals,
        'sig_uncorrected': sig_uncorrected,
        'cluster_sig_mask': cluster_sig_mask,
        'cluster_summary': cluster_summary,
        'n_participants': n_participants,
    }


def plot_group_decoding(times: np.ndarray,
                         all_mean_scores: np.ndarray,
                         stats_dict: dict,
                         out_path: Path) -> None:
    """
    Publication-style figure: group-mean AUC over time (+/- SEM shading),
    individual participant traces (thin, semi-transparent), chance level,
    and shaded regions where the cluster-based permutation test is
    significant (p < .05).
    """
    mean_auc = stats_dict['mean_auc']
    sem_auc  = stats_dict['sem_auc']
    cluster_mask = stats_dict['cluster_sig_mask']
    times_ms = times * 1000

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)


    for i in range(all_mean_scores.shape[0]):
        ax.plot(times_ms, all_mean_scores[i], color='grey', alpha=0.25,
                 linewidth=0.8, zorder=1)


    ax.plot(times_ms, mean_auc, color='#1f4e79', linewidth=2.2,
             label='Group mean AUC', zorder=3)
    ax.fill_between(times_ms, mean_auc - sem_auc, mean_auc + sem_auc,
                     color='#1f4e79', alpha=0.25, zorder=2,
                     label='+/- SEM')


    ax.axhline(CHANCE_LEVEL, color='black', linestyle='--', linewidth=1.0,
                label='Chance (AUC = 0.5)', zorder=1)
    ax.axvline(0, color='black', linestyle=':', linewidth=0.8, zorder=1)


    if np.any(cluster_mask):
        ax.fill_between(
            times_ms, 0, 1, where=cluster_mask, transform=ax.get_xaxis_transform(),
            color='#e07b39', alpha=0.20, zorder=0,
            label='Cluster p < .05',
        )

    ax.set_xlabel('Time (ms, relative to corrected sound onset)')
    ax.set_ylabel('Decoding performance (ROC-AUC)')
    ax.set_xlim(times_ms[0], times_ms[-1])
    ax.set_ylim(0.3, 1.0)
    ax.legend(loc='upper right', frameon=False, fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def run_decoding(participants: list[str], n_jobs: int = N_JOBS) -> None:
    DECODING_TABLE_PATH.mkdir(parents=True, exist_ok=True)
    DECODING_FIGURE_PATH.mkdir(parents=True, exist_ok=True)


    per_participant_results = {}
    all_mean_scores = []
    times_ref = None
    n_ok, n_skipped = 0, 0

    for pid in participants:
        try:
            epochs = load_looming_receding_epochs(pid)
            if epochs is None:
                n_skipped += 1
                continue

            result = decode_participant(epochs, n_jobs=n_jobs)
            if result is None:
                n_skipped += 1
                continue

            if times_ref is None:
                times_ref = result['times']
            elif not np.allclose(times_ref, result['times']):


                n_skipped += 1
                continue


            out_npy = DECODING_TABLE_PATH / f"{pid}_looming_vs_receding_scores.npy"
            np.save(out_npy, result['scores'])

            per_participant_results[pid] = {
                'n_looming': result['n_looming'],
                'n_receding': result['n_receding'],
                'peak_auc': float(result['mean'].max()),
                'peak_auc_time_ms': float(times_ref[np.argmax(result['mean'])] * 1000),
                'mean_auc_over_window': float(result['mean'].mean()),
            }

            all_mean_scores.append(result['mean'])
            n_ok += 1

        except Exception as e:

            traceback.print_exc()
            n_skipped += 1


    if n_ok < 2:


        return

    all_mean_scores = np.vstack(all_mean_scores)                             


    stats_dict = group_level_stats(all_mean_scores, times_ref)


    np.save(DECODING_TABLE_PATH / 'group_decoding_scores.npy', all_mean_scores)
    np.save(DECODING_TABLE_PATH / 'group_decoding_times.npy', times_ref)

    summary = {
        'n_participants': n_ok,
        'n_skipped': n_skipped,
        'decode_window_ms': [DECODE_TMIN * 1000, DECODE_TMAX * 1000],
        'classifier': 'linear SVM (C=1.0)',
        'cv': f'{N_SPLITS}-fold StratifiedKFold',
        'scoring': SCORING,
        'peak_group_auc': float(stats_dict['mean_auc'].max()),
        'peak_group_auc_time_ms': float(times_ref[np.argmax(stats_dict['mean_auc'])] * 1000),
        'cluster_permutation_summary': stats_dict['cluster_summary'],
        'per_participant': per_participant_results,
    }
    summary_path = DECODING_TABLE_PATH / 'group_decoding_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(_to_python(summary), f, indent=2, ensure_ascii=False)


    fig_path = DECODING_FIGURE_PATH / 'group_temporal_decoding.png'
    plot_group_decoding(times_ref, all_mean_scores, stats_dict, fig_path)


def main():
    parser = argparse.ArgumentParser(
        description='MVPA temporal decoding: looming vs. receding auditory '
                     'distance deviants (single-trial EEG, MMN window).'
    )
    parser.add_argument('--participants', nargs='+', default=cfg.PARTICIPANTS,
                         help='Participant IDs to include (default: all in config.py)')
    parser.add_argument('--n-jobs', type=int, default=1,
                         help='Number of parallel jobs for SlidingEstimator/CV')
    args = parser.parse_args()

    cfg.ensure_output_dirs()
    run_decoding(args.participants, n_jobs=args.n_jobs)


if __name__ == '__main__':
    main()
    sys.exit(0)