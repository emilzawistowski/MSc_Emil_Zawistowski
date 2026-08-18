import sys
import json
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import mne

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg              

warnings.filterwarnings('ignore', category=RuntimeWarning)
mne.set_log_level('ERROR')

OUT_TABLES = cfg.STATS_PATH
OUT_REPORTS = cfg.REPORTS_PATH
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_REPORTS.mkdir(parents=True, exist_ok=True)

LINE_FREQ = cfg.NOTCH_FREQ                
ALPHA_BAND = (8.0, 12.0)
LINE_BAND_HALFWIDTH = 1.0                                  

BLOCKS = ['A', 'B', 'C']
COND_LABELS = {'std': 'standard', 'dev': 'deviant'}


def find_epoch_files() -> dict:
    """Returns {participant: {block: {cond: Path}}} based on *_epo.fif files."""
    out = defaultdict(lambda: defaultdict(dict))
    for f in cfg.PREPROCESSED_PATH.glob("*_epo.fif"):

        stem = f.stem.replace("_epo", "")
        parts = stem.split("_")
        if len(parts) < 3:
            continue
        participant = parts[0]
        block_part = parts[1]            
        cond = parts[2]                       
        if not block_part.startswith("block"):
            continue
        block = block_part.replace("block", "")
        out[participant][block][cond] = f
    return out

def load_report_json(participant: str, block: str) -> dict:
    path = cfg.PREPROCESSED_PATH / f"{participant}_block{block}_report.json"
    if not path.exists():
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def psd_metrics(epochs: mne.Epochs) -> dict:
    """Computes PSD on epochs and extracts simple, interpretable indicators."""
    try:
        psd = epochs.compute_psd(method='welch', fmin=1, fmax=45, verbose=False)
        freqs = psd.freqs
        data = psd.get_data()                                   
        mean_psd = data.mean(axis=(0, 1))                                     

        def band_power(fmin, fmax):
            mask = (freqs >= fmin) & (freqs <= fmax)
            return float(mean_psd[mask].mean()) if mask.any() else np.nan

        line_power = band_power(LINE_FREQ - LINE_BAND_HALFWIDTH, LINE_FREQ + LINE_BAND_HALFWIDTH)
        neighbor_power = band_power(LINE_FREQ - 6, LINE_FREQ - 3)                           
        alpha_power = band_power(*ALPHA_BAND)
        broadband_power = band_power(3, 30)

        line_noise_ratio = float(line_power / neighbor_power) if neighbor_power > 0 else np.nan
        alpha_ratio = float(alpha_power / broadband_power) if broadband_power > 0 else np.nan

        return {
            'line_noise_ratio_50hz': line_noise_ratio,                                
            'alpha_power_ratio': alpha_ratio,                                                                                       
        }
    except Exception as e:
        return {'line_noise_ratio_50hz': np.nan, 'alpha_power_ratio': np.nan, 'psd_error': str(e)}

def channel_outlier_metrics(epochs: mne.Epochs) -> dict:
    """Checks if anything still stands out despite earlier bad‑channel detection."""
    data = epochs.get_data(picks='eeg')                                   
    ch_names = epochs.copy().pick('eeg').ch_names
    per_channel_var = data.var(axis=(0, 2))
    median_var = np.median(per_channel_var)
    mad = np.median(np.abs(per_channel_var - median_var)) + 1e-20
    robust_z = (per_channel_var - median_var) / (1.4826 * mad)

    still_outlying = [ch for ch, z in zip(ch_names, robust_z) if z > 4]
    return {
        'n_channels_still_outlying': len(still_outlying),
        'still_outlying_channels': ";".join(still_outlying),
        'median_channel_variance': float(median_var),
    }

def gfp_metrics(epochs: mne.Epochs) -> dict:
    evoked = epochs.average()
    gfp = evoked.data.std(axis=0)                                
    return {
        'gfp_mean': float(gfp.mean()),
        'gfp_peak': float(gfp.max()),
        'gfp_peak_latency_ms': float(evoked.times[np.argmax(gfp)] * 1000),
    }

def run_quality_audit(files_index: dict) -> pd.DataFrame:
    rows = []


    for participant, blocks in sorted(files_index.items()):
        for block, conds in sorted(blocks.items()):
            report = load_report_json(participant, block)
            for cond, fpath in conds.items():

                try:
                    epochs = mne.read_epochs(fpath, preload=True, verbose=False)
                except Exception as e:
                    rows.append({'participant': participant, 'block': block, 'cond': cond,
                                 'error': f"cannot load: {e}"})
                    continue

                row = {
                    'participant': participant, 'block': block, 'cond': cond,
                    'n_epochs_final': len(epochs),
                    'n_channels': len(epochs.ch_names),
                    'n_bad_channels_recorded': len(report.get('bad_channels', [])),
                    'bad_channels': ";".join(report.get('bad_channels', [])),
                    'n_bridged_pairs': len(report.get('bridged_electrode_pairs', [])),
                    'n_ica_removed': len(report.get('ica_components_removed', [])),
                    'pct_rejected': report.get(f'pct_{cond}_rejected', np.nan),
                }
                row.update(psd_metrics(epochs))
                row.update(channel_outlier_metrics(epochs))
                row.update(gfp_metrics(epochs))
                rows.append(row)

    df = pd.DataFrame(rows)
    out_path = OUT_TABLES / 'quality_audit.csv'
    df.to_csv(out_path, index=False)

    return df


def roi_amplitude(evoked: mne.Evoked, roi: list[str], tmin: float, tmax: float) -> float:
    picks = [ch for ch in roi if ch in evoked.ch_names]
    if not picks:
        return np.nan
    data = evoked.copy().pick(picks).get_data(tmin=tmin, tmax=tmax)
    return float(data.mean() * 1e6)         

def baseline_check(evoked: mne.Evoked, roi: list[str]) -> float:
    picks = [ch for ch in roi if ch in evoked.ch_names]
    if not picks:
        return np.nan
    data = evoked.copy().pick(picks).get_data(tmin=cfg.BASELINE[0], tmax=cfg.BASELINE[1])
    return float(data.mean() * 1e6)

def erp_snr(evoked: mne.Evoked, roi: list[str], tmin: float, tmax: float) -> float:
    """SNR = |amplitude in effect window| / SD of signal in baseline window."""
    picks = [ch for ch in roi if ch in evoked.ch_names]
    if not picks:
        return np.nan
    effect = evoked.copy().pick(picks).get_data(tmin=tmin, tmax=tmax).mean(axis=0)
    baseline = evoked.copy().pick(picks).get_data(tmin=cfg.BASELINE[0], tmax=cfg.BASELINE[1]).mean(axis=0)
    noise_sd = baseline.std()
    if noise_sd == 0:
        return np.nan
    return float(np.abs(effect).mean() / noise_sd)

def run_erp_diagnostics(files_index: dict) -> pd.DataFrame:
    rows = []


    for participant, blocks in sorted(files_index.items()):
        for block, conds in sorted(blocks.items()):
            if 'std' not in conds or 'dev' not in conds:
                continue                                                        
            try:
                ep_std = mne.read_epochs(conds['std'], preload=True, verbose=False)
                ep_dev = mne.read_epochs(conds['dev'], preload=True, verbose=False)
            except Exception as e:

                continue

            ev_std = ep_std.average()
            ev_dev = ep_dev.average()


            ev_diff = mne.combine_evoked([ev_dev, ev_std], weights=[1, -1])

            mmn_amp = roi_amplitude(ev_diff, cfg.ROI_MMN, cfg.MMN_TMIN, cfg.MMN_TMAX)
            p3a_amp = roi_amplitude(ev_diff, cfg.ROI_P3A, cfg.P3A_TMIN, cfg.P3A_TMAX)

            row = {
                'participant': participant, 'block': block,
                'n_std': len(ep_std), 'n_dev': len(ep_dev),
                'mmn_amplitude_uV': mmn_amp,
                'mmn_polarity': 'negative (expected)' if mmn_amp < 0 else 'POSITIVE (unusual!)',
                'p3a_amplitude_uV': p3a_amp,
                'p3a_polarity': 'positive (expected)' if p3a_amp > 0 else 'NEGATIVE (unusual!)',
                'mmn_snr': erp_snr(ev_diff, cfg.ROI_MMN, cfg.MMN_TMIN, cfg.MMN_TMAX),
                'p3a_snr': erp_snr(ev_diff, cfg.ROI_P3A, cfg.P3A_TMIN, cfg.P3A_TMAX),
                'baseline_mean_uV_std_cond': baseline_check(ev_std, cfg.ROI_MMN),
                'baseline_mean_uV_dev_cond': baseline_check(ev_dev, cfg.ROI_MMN),
                'custom_ref_applied': bool(ep_std.info.get('custom_ref_applied', False)),
            }
            rows.append(row)


    df = pd.DataFrame(rows)
    out_path = OUT_TABLES / 'erp_diagnostics.csv'
    df.to_csv(out_path, index=False)

    return df


def write_summary_report(df_quality: pd.DataFrame, df_erp: pd.DataFrame) -> None:
    lines = []
    lines.append("=" * 78)
    lines.append("  SIGNAL DIAGNOSTICS - SUMMARY REPORT")
    lines.append("=" * 78)


    lines.append("\n--- SIGNAL QUALITY ---")
    if not df_quality.empty:
        high_line_noise = df_quality[df_quality['line_noise_ratio_50hz'] > 4]
        lines.append(f"Files with visible 50 Hz peak (ratio > 4x background): {len(high_line_noise)} / {len(df_quality)}")
        if len(high_line_noise):
            for _, r in high_line_noise.iterrows():
                lines.append(f"  -> {r['participant']} block {r['block']} [{r['cond']}]: "
                              f"ratio={r['line_noise_ratio_50hz']:.1f}")

        still_bad = df_quality[df_quality['n_channels_still_outlying'] > 0]
        lines.append(f"\nFiles with channels STILL outlying AFTER preprocessing: {len(still_bad)} / {len(df_quality)}")
        if len(still_bad):
            for _, r in still_bad.iterrows():
                lines.append(f"  -> {r['participant']} block {r['block']} [{r['cond']}]: "
                              f"{r['still_outlying_channels']}")

        high_rejection = df_quality[df_quality['pct_rejected'] > 30]
        lines.append(f"\nFiles with >30% rejected epochs: {len(high_rejection)} / {len(df_quality)}")
        if len(high_rejection):
            for _, r in high_rejection.iterrows():
                lines.append(f"  -> {r['participant']} block {r['block']} [{r['cond']}]: "
                              f"{r['pct_rejected']:.1f}% rejected")


    lines.append("\n--- ERP: POLARITY AND SNR ---")
    if not df_erp.empty:
        n_mmn_pos = (df_erp['mmn_amplitude_uV'] >= 0).sum()
        n_total = len(df_erp)

        lines.append(f"MMN negative (convention‑compliant): {n_total - n_mmn_pos} / {n_total}")
        lines.append(f"MMN positive (UNUSUAL):               {n_mmn_pos} / {n_total}")

        from scipy import stats as _stats
        binom_p = _stats.binomtest(int(n_mmn_pos), n_total, 0.5).pvalue
        lines.append(f"Binomial test vs 50/50 (whether deviation from chance is significant): p = {binom_p:.3f}")

        if binom_p < 0.05 and n_mmn_pos > n_total / 2:
            lines.append(
                "\n  >> The proportion of positive MMNs is STATISTICALLY SIGNIFICANTLY higher than chance (p<0.05). "
                "This IS evidence for a systematic problem (not just noise). Check in order:"
            )
            lines.append("     1. Are 'dev' and 'std' definitely not swapped in event_id/trigger_remap?")
            lines.append("     2. Subtraction order in combine_evoked (here: dev - std)")
            lines.append("     3. Reference (average vs mastoid)")
            lines.append("     4. Amplifier/electrode polarity")
        elif binom_p < 0.05 and n_mmn_pos < n_total / 2:
            lines.append("\n  >> Proportion of NEGATIVE MMNs is significantly higher than chance - this matches convention, OK.")
        else:
            lines.append(
                f"\n  >> Deviation from 50/50 is NOT statistically significant (p={binom_p:.2f}). "
                f"This is consistent with the hypothesis that the true MMN effect is close to zero/noise in this sample "
                f"(consistent with BF10<1 from 03_statistics.py), and NOT a systematic polarity error. "
                f"Individual blocks with positive MMN below are worth reviewing per participant, "
                f"but they do not indicate a pipeline‑wide mistake."
            )

        low_snr = df_erp[(df_erp['mmn_snr'] < 1.0)]
        lines.append(f"\nBlocks with low SNR for MMN (<1.0): {len(low_snr)} / {len(df_erp)}")

        bad_baseline = df_erp[df_erp['baseline_mean_uV_std_cond'].abs() > 1.0]
        if len(bad_baseline):
            lines.append(f"\n  WARNING: {len(bad_baseline)} blocks have mean baseline > 1 uV away from zero "
                          f"- baseline correction may not work properly on these files:")
            for _, r in bad_baseline.iterrows():
                lines.append(f"     - {r['participant']} block {r['block']}: "
                              f"baseline_std={r['baseline_mean_uV_std_cond']:.2f} uV")

    lines.append("\n" + "=" * 78)
    report_text = "\n".join(lines)


    out_path = OUT_REPORTS / 'signal_diagnostics.txt'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(report_text)


def main():

    files_index = find_epoch_files()
    n_files = sum(len(c) for b in files_index.values() for c in b.values())


    if n_files == 0:

        sys.exit(1)

    df_quality = run_quality_audit(files_index)
    df_erp = run_erp_diagnostics(files_index)
    write_summary_report(df_quality, df_erp)

if __name__ == '__main__':
    main()