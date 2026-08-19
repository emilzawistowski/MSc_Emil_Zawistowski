import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import mne

import shutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg  # noqa: E402

warnings.filterwarnings('ignore', category=RuntimeWarning)
mne.set_log_level('ERROR')

OUT_PATH = cfg.STATS_PATH / 'raw_noise_check.csv'
TMP_DIR = cfg.TMP_PATH / 'raw_noise_check_tmp'


def _read_internal_filenames(vhdr_path: Path) -> dict:
    """Reads DataFile=/MarkerFile= entries from a .vhdr header."""
    result = {'eeg': None, 'vmrk': None}
    try:
        with open(vhdr_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                s = line.strip()
                if s.lower().startswith('datafile='):
                    result['eeg'] = s.split('=', 1)[1].strip()
                elif s.lower().startswith('markerfile='):
                    result['vmrk'] = s.split('=', 1)[1].strip()
                if result['eeg'] and result['vmrk']:
                    break
    except OSError as e:
        print(f"    WARNING: Could not read header from {vhdr_path.name}: {e}")
    return result


def copy_files_to_tmp(files_dict: dict, tmp_dir: Path) -> Path:
    """Copies .vhdr, .eeg and the cleaned .vmrk into one directory so
    mne.io.read_raw_brainvision can load them together."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    vhdr_src = files_dict['vhdr']

    internal = _read_internal_filenames(vhdr_src)
    eeg_internal = internal['eeg'] or (vhdr_src.stem + '.eeg')
    vmrk_internal = internal['vmrk'] or (vhdr_src.stem + '.vmrk')

    vhdr_dst = tmp_dir / vhdr_src.name
    shutil.copy2(vhdr_src, vhdr_dst)

    eeg_src = files_dict.get('eeg')
    if eeg_src and eeg_src.exists():
        shutil.copy2(eeg_src, tmp_dir / eeg_internal)
    else:
        raise FileNotFoundError(f"Missing .eeg for {vhdr_src.stem}")

    cleaned_vmrk = files_dict.get('vmrk_cleaned')
    if cleaned_vmrk and cleaned_vmrk.exists():
        shutil.copy2(cleaned_vmrk, tmp_dir / vmrk_internal)
    else:
        raise FileNotFoundError(f"Missing cleaned .vmrk for {vhdr_src.stem}")

    return vhdr_dst


def cleanup_tmp(tmp_dir: Path, vhdr_name: str) -> None:
    stem = Path(vhdr_name).stem
    for ext in ('.vhdr', '.eeg', '.vmrk'):
        f = tmp_dir / f"{stem}{ext}"
        if f.exists():
            f.unlink()


def analyze_raw_noise(raw: mne.io.Raw) -> dict:
    """Computes PSD on unfiltered raw data and extracts noise indicators."""
    data = raw.get_data(picks='eeg')
    sfreq = raw.info['sfreq']

    psd = raw.copy().pick('eeg').compute_psd(method='welch', fmin=0.1, fmax=min(sfreq / 2 - 1, 200),
                                              verbose=False)
    freqs = psd.freqs
    mean_psd = psd.get_data().mean(axis=0)

    def band_power(fmin, fmax):
        mask = (freqs >= fmin) & (freqs <= fmax)
        return float(mean_psd[mask].mean()) if mask.any() else np.nan

    signal_band = band_power(1, 30)
    line_50 = band_power(49, 51)
    line_100 = band_power(99, 101)
    line_150 = band_power(149, 151)
    neighbor_50 = band_power(45, 48)
    broadband_high = band_power(60, min(sfreq / 2 - 5, 190))
    dc_drift = band_power(0.1, 0.5)

    line_ratio_50 = line_50 / neighbor_50 if neighbor_50 > 0 else np.nan

    log_psd = np.log10(mean_psd + 1e-30)
    baseline = np.median(log_psd)
    mad = np.median(np.abs(log_psd - baseline)) + 1e-12
    z = (log_psd - baseline) / (1.4826 * mad)

    known_freqs = [50, 100, 150]
    unexplained_peaks = []
    peak_idx = np.where(z > 6)[0]
    for idx in peak_idx:
        f = freqs[idx]
        if not any(abs(f - kf) < 1.5 for kf in known_freqs):
            unexplained_peaks.append(round(float(f), 1))

    return {
        'signal_band_power_1_30hz': signal_band,
        'line_power_50hz': line_50,
        'line_ratio_50hz_vs_neighbor': line_ratio_50,
        'line_power_100hz': line_100,
        'line_power_150hz': line_150,
        'broadband_noise_60_190hz': broadband_high,
        'broadband_vs_signal_ratio': broadband_high / signal_band if signal_band > 0 else np.nan,
        'dc_drift_power_0.1_0.5hz': dc_drift,
        'unexplained_narrow_peaks_hz': ";".join(map(str, unexplained_peaks)) if unexplained_peaks else "",
    }


def drift_over_time(raw: mne.io.Raw) -> dict:
    """Splits the recording into 4 quarters and compares RMS across them."""
    data = raw.get_data(picks='eeg')
    n_times = data.shape[1]
    quarter = n_times // 4
    rms_per_quarter = []
    for q in range(4):
        seg = data[:, q * quarter:(q + 1) * quarter]
        rms_per_quarter.append(float(np.sqrt(np.mean(seg ** 2))))

    trend_ratio = rms_per_quarter[-1] / rms_per_quarter[0] if rms_per_quarter[0] > 0 else np.nan
    return {
        'rms_q1': rms_per_quarter[0], 'rms_q2': rms_per_quarter[1],
        'rms_q3': rms_per_quarter[2], 'rms_q4': rms_per_quarter[3],
        'rms_last_vs_first_ratio': trend_ratio,
    }


def main():
    rows = []
    for participant_id in cfg.PARTICIPANTS:
        files = cfg.get_participant_files(participant_id)
        for block_type, files_dict in files.items():
            vhdr = files_dict.get('vhdr')
            if vhdr is None:
                continue
            print(f"{participant_id} block {block_type}: {vhdr.name}")
            try:
                tmp_vhdr = copy_files_to_tmp(files_dict, TMP_DIR)
                raw = mne.io.read_raw_brainvision(str(tmp_vhdr), preload=True, verbose=False)
            except Exception as e:
                print(f"  ERROR loading: {e}")
                continue
            finally:
                cleanup_tmp(TMP_DIR, vhdr.name)

            row = {'participant': participant_id, 'block': block_type}
            row.update(analyze_raw_noise(raw))
            row.update(drift_over_time(raw))
            rows.append(row)

    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")

    if df.empty:
        print("\nNo files were loaded correctly - check errors above.")
        return

    print("\n--- SUMMARY ---")
    high_line = df[df['line_ratio_50hz_vs_neighbor'] > 3]
    print(f"Files with visible 50Hz peak in raw data (ratio>3): {len(high_line)} / {len(df)}")

    with_unexplained = df[df['unexplained_narrow_peaks_hz'] != ""]
    print(f"Files with unexplained narrow peaks (not 50/100/150Hz): {len(with_unexplained)} / {len(df)}")
    for _, r in with_unexplained.iterrows():
        print(f"  -> {r['participant']} block {r['block']}: {r['unexplained_narrow_peaks_hz']} Hz")

    rising_noise = df[df['rms_last_vs_first_ratio'] > 1.5]
    print(f"\nFiles with RMS increase >50% in the second half of the recording: {len(rising_noise)} / {len(df)}")
    for _, r in rising_noise.iterrows():
        print(f"  -> {r['participant']} block {r['block']}: ratio={r['rms_last_vs_first_ratio']:.2f}")

    high_dc = df.sort_values('dc_drift_power_0.1_0.5hz', ascending=False).head(5)
    print(f"\nTop 5 recordings with highest DC drift (0.1-0.5Hz):")
    for _, r in high_dc.iterrows():
        print(f"  -> {r['participant']} block {r['block']}: {r['dc_drift_power_0.1_0.5hz']:.3e}")


if __name__ == '__main__':
    main()