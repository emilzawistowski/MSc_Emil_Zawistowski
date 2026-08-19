import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ALPHA = 0.05
TARGET_POWER = 0.80


def _power_one_sample(d: float, n: int, alpha: float, alternative: str) -> float:
    """Analytic power for a one-sample/paired t-test at a given Cohen's d,
    via the noncentral t-distribution."""
    dof = n - 1
    ncp = d * np.sqrt(n)
    if alternative == "one-sided":
        t_crit = stats.t.ppf(1 - alpha, dof)
        power = 1 - stats.nct.cdf(t_crit, dof, ncp)
    else:
        t_crit = stats.t.ppf(1 - alpha / 2, dof)
        power = 1 - stats.nct.cdf(t_crit, dof, ncp) + stats.nct.cdf(-t_crit, dof, ncp)
    return float(power)


def minimum_detectable_d(
    n: int,
    alpha: float = ALPHA,
    target_power: float = TARGET_POWER,
    alternative: str = "one-sided",
    d_max: float = 3.0,
) -> float:
    """Smallest |Cohen's d| for which power(d) >= target_power at this N,
    found via bisection."""
    lo, hi = 1e-4, d_max
    if _power_one_sample(hi, n, alpha, alternative) < target_power:
        log.warning("d_max=%.2f insufficient for N=%d to reach power=%.2f; "
                     "returned value is a lower bound only.", d_max, n, target_power)
        return hi
    for _ in range(100):
        mid = (lo + hi) / 2
        if _power_one_sample(mid, n, alpha, alternative) < target_power:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2)


def achieved_power(observed_d: float, n: int, alpha: float = ALPHA,
                    alternative: str = "one-sided") -> float:
    """Post-hoc power at the observed effect size (reported for context only)."""
    return _power_one_sample(abs(observed_d), n, alpha, alternative)


def load_erp_measures() -> pd.DataFrame:
    csv_path = Path(config.STATS_PATH) / "erp_measures.csv"
    df = pd.read_csv(csv_path)
    log.info("Loaded %d rows from %s.", len(df), csv_path)
    return df


def load_descriptives() -> pd.DataFrame | None:
    csv_path = Path(config.STATS_PATH) / "descriptive_statistics.csv"
    if not csv_path.exists():
        log.warning("descriptive_statistics.csv not found; run 03_statistics.py first.")
        return None
    return pd.read_csv(csv_path)


TESTS = [
    ("H1: MMN_A (Near-deviant, block A) vs 0", "mmn_A_amp_Fz", "one-sample", "one-sided"),
    ("H1: MMN_B (Far-deviant, block B) vs 0", "mmn_B_amp_Fz", "one-sample", "one-sided"),
    ("H3: P3a_A (deviant block A) vs 0", "p3a_A_amp_Cz", "one-sample", "one-sided"),
    ("H3: P3a_B (deviant block B) vs 0", "p3a_B_amp_Cz", "one-sample", "one-sided"),
    ("SSA_AC: dev_A - ctrl_C vs 0", "ssa_AC_amp_Fz", "one-sample", "one-sided"),
    ("SSA_BC: dev_B - ctrl_C vs 0", "ssa_BC_amp_Fz", "one-sample", "one-sided"),
]


def sensitivity_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for label, col, kind, alt in TESTS:
        series = df[col].dropna() if col in df.columns else pd.Series(dtype=float)
        n = len(series)
        if n < 3:
            log.warning("Skipping '%s': N=%d too small.", label, n)
            continue

        mde_d = minimum_detectable_d(n, ALPHA, TARGET_POWER, alt)
        sd = float(series.std(ddof=1))
        observed_mean = float(series.mean())
        observed_d = observed_mean / sd if sd > 0 else np.nan
        mde_uv = mde_d * sd

        rows.append({
            "Test": label,
            "N": n,
            "Alternative": alt,
            "Observed_SD_uV": round(sd, 4),
            "Observed_d": round(observed_d, 3) if not np.isnan(observed_d) else np.nan,
            "MDE_Cohen_d_80pct_power": round(mde_d, 3),
            "MDE_uV_at_observed_SD": round(mde_uv, 3),
            "Achieved_power_at_observed_d": round(achieved_power(observed_d, n, ALPHA, alt), 3)
                if not np.isnan(observed_d) else np.nan,
        })

    common_idx = df["mmn_A_amp_Fz"].notna() & df["mmn_B_amp_Fz"].notna()
    diff = df.loc[common_idx, "mmn_A_amp_Fz"] - df.loc[common_idx, "mmn_B_amp_Fz"]
    n = len(diff)
    if n >= 3:
        sd = float(diff.std(ddof=1))
        mde_d = minimum_detectable_d(n, ALPHA, TARGET_POWER, "one-sided")
        observed_mean = float(diff.mean())
        observed_d = observed_mean / sd if sd > 0 else np.nan
        rows.append({
            "Test": "H2: MMN_A vs MMN_B (paired, A < B)",
            "N": n,
            "Alternative": "one-sided",
            "Observed_SD_uV": round(sd, 4),
            "Observed_d": round(observed_d, 3) if not np.isnan(observed_d) else np.nan,
            "MDE_Cohen_d_80pct_power": round(mde_d, 3),
            "MDE_uV_at_observed_SD": round(mde_d * sd, 3),
            "Achieved_power_at_observed_d": round(achieved_power(observed_d, n, ALPHA, "one-sided"), 3)
                if not np.isnan(observed_d) else np.nan,
        })

    return pd.DataFrame(rows)


def generate_report(table: pd.DataFrame) -> str:
    lines = [
        "=" * 70,
        "SENSITIVITY (POST-HOC) POWER ANALYSIS",
        "=" * 70,
        "",
        f"alpha = {ALPHA}, target power = {TARGET_POWER}",
        "",
        "This is a sensitivity analysis, not a post-hoc power analysis on the",
        "observed effect (see Hoenig & Heisey, 2001). It reports the smallest",
        "true effect (Cohen's d) this design had an 80% chance of detecting.",
        "",
        table.to_string(index=False),
        "",
        "=" * 70,
        "READING GUIDE:",
        "  - MDE_Cohen_d_80pct_power: minimum detectable effect at N and alpha.",
        "  - MDE_uV_at_observed_SD: the same MDE in microvolts, using this",
        "    test's observed between-subject SD.",
        "  - Observed_d / Achieved_power_at_observed_d are reported for",
        "    context only; MDE is the primary sensitivity statistic.",
        "=" * 70,
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    log.info("=" * 60)
    log.info("07_power_analysis.py -- START")
    log.info("=" * 60)

    Path(config.STATS_PATH).mkdir(parents=True, exist_ok=True)
    df = load_erp_measures()

    table = sensitivity_table(df)
    out_csv = Path(config.STATS_PATH) / "sensitivity_power_analysis.csv"
    table.to_csv(out_csv, index=False)
    log.info("Saved: %s", out_csv)

    report = generate_report(table)
    out_txt = Path(config.STATS_PATH) / "sensitivity_power_analysis.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(report)
    log.info("Saved: %s", out_txt)
    print("\n" + report)

    log.info("=" * 60)
    log.info("07_power_analysis.py -- FINISHED SUCCESSFULLY")
    log.info("=" * 60)


if __name__ == "__main__":
    main()