import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg              

IN_PATH = cfg.STATS_PATH / 'raw_noise_check.csv'


def _robust_z(values: pd.Series) -> pd.Series:
    center = np.median(values)
    mad = np.median(np.abs(values - center)) + 1e-12
    return (values - center) / (1.4826 * mad)


def flag_recordings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['flag_rising_noise'] = (
        df['rms_last_vs_first_ratio'] > cfg.RISING_NOISE_RMS_RATIO_THRESHOLD
    )

    log_power = np.log10(df['signal_band_power_1_30hz'].astype(float) + 1e-30)
    df['signal_power_robust_z'] = _robust_z(log_power)
    df['flag_anomalous_power'] = (
        df['signal_power_robust_z'].abs() > cfg.SIGNAL_POWER_ROBUST_Z_THRESHOLD
    )

    q_cols = ['rms_q2', 'rms_q3', 'rms_q4']
    q_mean = df[q_cols].mean(axis=1)
    q_std = df[q_cols].std(axis=1, ddof=1)
    df['flat_rms_rel_std'] = q_std / q_mean.replace(0, np.nan)
    df['flag_flat_clipped'] = (
        df['flat_rms_rel_std'] < cfg.FLAT_RMS_REL_STD_THRESHOLD
    )

    df['excluded'] = (
        df['flag_rising_noise'] | df['flag_anomalous_power'] | df['flag_flat_clipped']
    )

    def _reasons(row) -> str:
        reasons = []
        if row['flag_rising_noise']:
            reasons.append(
                f"rising_noise(rms_ratio={row['rms_last_vs_first_ratio']:.2f}"
                f">{cfg.RISING_NOISE_RMS_RATIO_THRESHOLD})"
            )
        if row['flag_anomalous_power']:
            reasons.append(
                f"anomalous_signal_power(z={row['signal_power_robust_z']:.1f}, "
                f"threshold=|{cfg.SIGNAL_POWER_ROBUST_Z_THRESHOLD}|)"
            )
        if row['flag_flat_clipped']:
            reasons.append(
                f"flat_clipped_signal(rel_std={row['flat_rms_rel_std']:.4f}"
                f"<{cfg.FLAT_RMS_REL_STD_THRESHOLD})"
            )
        return "; ".join(reasons)

    df['reasons'] = df.apply(_reasons, axis=1)
    return df


def main():
    if not IN_PATH.exists():

        sys.exit(1)

    df = pd.read_csv(IN_PATH)


    flagged = flag_recordings(df)
    excluded = flagged[flagged['excluded']].copy()

    out_cols = [
        'participant', 'block', 'reasons',
        'rms_last_vs_first_ratio', 'signal_power_robust_z', 'flat_rms_rel_std',
    ]
    excluded_out = excluded[out_cols].sort_values(['participant', 'block'])

    cfg.STATS_PATH.mkdir(parents=True, exist_ok=True)
    excluded_out.to_csv(cfg.EXCLUDED_RECORDINGS_PATH, index=False)


    if len(excluded_out):

        pass
    else:


        pass
if __name__ == '__main__':
    main()