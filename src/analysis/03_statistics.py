import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mne
from scipy import integrate, stats
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
mne.set_log_level("WARNING")


def _print_section(title: str) -> None:
    border = "═" * 60
    log.info("\n%s\n  %s\n%s", border, title, border)


def _interpret_bf(bf10: float, threshold: float = None) -> str:
    thr = threshold or getattr(config, "BF10_THRESHOLD", 3.0)
    if pd.isna(bf10):
        return "BF10: no data"
    if bf10 > 100:
        return f"BF10 = {bf10:.2f} → very strong evidence for H1"
    if bf10 > 30:
        return f"BF10 = {bf10:.2f} → strong evidence for H1"
    if bf10 > thr:
        return f"BF10 = {bf10:.2f} → moderate evidence for H1 (threshold: {thr})"
    if bf10 > 1:
        return f"BF10 = {bf10:.2f} → weak evidence for H1 (< threshold {thr})"
    if bf10 > 1 / thr:
        return f"BF10 = {bf10:.2f} → no evidence"
    return f"BF10 = {bf10:.2f} → evidence for H0"


def jzs_bayes_factor(t: float, n: int, r_scale: float = 0.5) -> float:
    """JZS Bayes Factor (BF10) for a t-test with a Cauchy prior (scale
    r_scale) on the effect size, per Rouder et al. (2009) / Wagenmakers
    (2007), evaluated by numerical integration over an inverse-gamma(1/2,
    1/2) prior on g."""
    if n < 2:
        return np.nan
    df = n - 1

    def integrand(g):
        term1 = (1 + n * g) ** (-0.5)
        term2 = (1 + t**2 / ((1 + n * g) * df)) ** (-n / 2)
        prior = (0.5**0.5) / np.sqrt(np.pi) * g ** (-1.5) * np.exp(-1 / (2 * g))
        return term1 * term2 * prior

    try:
        result, _ = integrate.quad(integrand, 0, 1000, limit=1000)
        return float(result)
    except Exception:
        return np.nan


def _safe_ttest(
    data: pd.Series,
    y: float | pd.Series = 0.0,
    paired: bool = False,
    alternative: str = "two-sided",
    label: str = "",
    r_scale: float = 0.5,
) -> dict | None:
    """One-sample/paired t-test via scipy.stats plus a JZS Bayes Factor."""
    try:
        clean = data.dropna()
        if isinstance(y, pd.Series):
            y_clean = y.dropna()
            valid_idx = clean.index.intersection(y_clean.index)
            clean, y_clean = clean[valid_idx], y_clean[valid_idx]
            n = len(clean)
        else:
            y_clean = y
            n = len(clean)

        if n < 3:
            log.warning("Not enough observations for '%s': N=%d.", label, n)
            return None

        if paired:
            diff = clean - y_clean
            t_stat, p_val = stats.ttest_1samp(diff, 0, alternative=alternative)
            dof = n - 1
            mean_diff = diff.mean()
            sd_diff = diff.std(ddof=1)
            d = mean_diff / sd_diff if sd_diff != 0 else np.nan
            se = sd_diff / np.sqrt(n)
            if se > 0:
                crit = stats.t.ppf(1 - 0.025, df=dof)
                ci = (mean_diff - crit * se, mean_diff + crit * se)
            else:
                ci = None
            mean_val = clean.mean()
        else:
            t_stat, p_val = stats.ttest_1samp(clean, y_clean, alternative=alternative)
            dof = n - 1
            mean_diff = clean.mean() - y_clean
            sd = clean.std(ddof=1)
            d = t_stat / np.sqrt(n)  # Cohen's d for one-sample
            se = sd / np.sqrt(n)
            if se > 0:
                crit = stats.t.ppf(1 - 0.025, df=dof)
                ci = (mean_diff - crit * se, mean_diff + crit * se)
            else:
                ci = None
            mean_val = clean.mean()

        bf10 = jzs_bayes_factor(t_stat, n, r_scale)

        log.info("t-test '%s': N=%d, mean=%.3f, t=%.3f, p=%.4f, BF10=%.3f, d=%.3f",
                 label, n, float(mean_val), t_stat, p_val, bf10, d)

        return {
            "T": t_stat,
            "dof": dof,
            "p-val": p_val,
            "cohen-d": d,
            "BF10": bf10,
            "CI95%": ci,
            "mean_diff": mean_diff,
            "n": n,
            "mean": float(mean_val),
            "sd": float(clean.std(ddof=1)) if not paired else float(sd_diff),
        }
    except Exception as exc:
        log.error("Error in t-test '%s': %s", label, exc)
        return None


def _format_ieee_ttest(
    result: dict | None,
    mean: float,
    sd: float,
    label: str,
    note: str = "",
    unit: str = "µV",
) -> str:
    """Generates an IEEE-styled string based on a results dictionary."""
    if result is None:
        return f"{label}: Insufficient data for analysis.\n"

    T = result["T"]
    df = result["dof"]
    p = result["p-val"]
    d = result["cohen-d"]
    bf = result["BF10"]
    ci = result["CI95%"]
    n = result["n"]

    try:
        bf_val = float(bf)
        bf_str = f"{bf_val:.2f}" if not np.isnan(bf_val) else "NaN"
    except (ValueError, TypeError):
        bf_str = str(bf)

    p_str = f"= {p:.3f}" if p >= 0.001 else "< 0.001"
    ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "N/A"

    lines = [
        f"--- {label} ---",
        f"Descriptive Statistics: N = {n}, M = {mean:.3f} {unit}, SD = {sd:.3f} {unit}",
        f"Inferential Statistics: t({df:.1f}) = {T:.3f}, p {p_str}",
        f"Effect Size & Bayes Factor: Cohen's d = {d:.3f}, BF10 = {bf_str}",
        f"95% Confidence Interval: {ci_str}",
    ]
    if note:
        lines.append(f"Note: {note}")
    lines.append("")
    return "\n".join(lines)


# DATA LOADING
def load_erp_measures() -> pd.DataFrame:
    csv_path = Path(config.STATS_PATH) / "erp_measures.csv"
    try:
        df = pd.read_csv(csv_path)
        log.info("Loaded %d rows from %s.", len(df), csv_path)
        return df
    except FileNotFoundError:
        log.error("File does not exist: %s\nRun 02_erp_analysis.py first.", csv_path)
        raise
    except Exception as exc:
        log.error("Error loading CSV: %s", exc)
        raise


# ANALYSIS 1: H1 – MMN Significance
def analysis_h1(df: pd.DataFrame) -> dict:
    _print_section("ANALYSIS 1: H1 – MMN Significance (Bayesian t-test vs 0)")
    results = {}
    for key, col in [("mmn_A", "mmn_A_amp_Fz"), ("mmn_B", "mmn_B_amp_Fz")]:
        amps = df[col].dropna()
        n = len(amps)
        m = float(amps.mean())
        sd = float(amps.std(ddof=1))
        log.info("\n[H1] %s: N=%d, M=%.3f µV, SD=%.3f µV", key.upper(), n, m, sd)
        result = _safe_ttest(amps, 0.0, paired=False, alternative="less", label=f"H1_{key}")
        results[key] = {"result": result, "n": n, "mean": m, "sd": sd}
        if result is not None:
            bf_val = result.get("BF10", np.nan)
            if not np.isnan(bf_val):
                log.info(_interpret_bf(bf_val, getattr(config, "BF10_THRESHOLD", 3.0)))
    return results


# ANALYSIS 2: H2 – Looming asymmetry (DIRECTIONAL: Near > Far)
def analysis_h2(df: pd.DataFrame) -> dict:
    _print_section("ANALYSIS 2: H2 – Looming asymmetry (A vs B, paired, DIRECTIONAL)")
    results = {}

    # --- Amplitude ---
    common_idx = df["mmn_A_amp_Fz"].notna() & df["mmn_B_amp_Fz"].notna()
    amp_A = df.loc[common_idx, "mmn_A_amp_Fz"]
    amp_B = df.loc[common_idx, "mmn_B_amp_Fz"]
    n_amp = int(common_idx.sum())
    log.info("[H2] Amplitude: N=%d pairs", n_amp)
    log.info("  MMN_A: M=%.3f, SD=%.3f", float(amp_A.mean()), float(amp_A.std(ddof=1)))
    log.info("  MMN_B: M=%.3f, SD=%.3f", float(amp_B.mean()), float(amp_B.std(ddof=1)))
    res_amp = _safe_ttest(amp_A, amp_B, paired=True, alternative="less", label="H2_amplitude")
    results["amplitude"] = {
        "result": res_amp,
        "n": n_amp,
        "mean_A": float(amp_A.mean()),
        "mean_B": float(amp_B.mean()),
        "sd_A": float(amp_A.std(ddof=1)),
        "sd_B": float(amp_B.std(ddof=1)),
    }

    # --- FAL ---
    common_fal = df["mmn_A_fal_ms"].notna() & df["mmn_B_fal_ms"].notna()
    fal_A = df.loc[common_fal, "mmn_A_fal_ms"]
    fal_B = df.loc[common_fal, "mmn_B_fal_ms"]
    n_fal = int(common_fal.sum())
    log.info("[H2] FAL: N=%d pairs", n_fal)
    log.info("  FAL_A: M=%.1f ms, SD=%.1f ms", float(fal_A.mean()), float(fal_A.std(ddof=1)))
    log.info("  FAL_B: M=%.1f ms, SD=%.1f ms", float(fal_B.mean()), float(fal_B.std(ddof=1)))
    res_fal = _safe_ttest(fal_A, fal_B, paired=True, alternative="less", label="H2_fal")
    results["fal"] = {
        "result": res_fal,
        "n": n_fal,
        "mean_A": float(fal_A.mean()) if n_fal > 0 else np.nan,
        "mean_B": float(fal_B.mean()) if n_fal > 0 else np.nan,
        "sd_A": float(fal_A.std(ddof=1)) if n_fal > 0 else np.nan,
        "sd_B": float(fal_B.std(ddof=1)) if n_fal > 0 else np.nan,
    }
    return results


# ANALYSIS 3: H3 – P3a Significance
def analysis_h3(df: pd.DataFrame) -> dict:
    _print_section("ANALYSIS 3: H3 – P3a Significance (t-test vs 0, alternative='greater')")
    results = {}
    for key, col in [("p3a_A", "p3a_A_amp_Cz"), ("p3a_B", "p3a_B_amp_Cz")]:
        amps = df[col].dropna()
        n = len(amps)
        m = float(amps.mean()) if n > 0 else np.nan
        sd = float(amps.std(ddof=1)) if n > 1 else np.nan
        log.info("[H3] %s: N=%d, M=%.3f µV, SD=%.3f µV", key.upper(), n, m, sd)
        result = _safe_ttest(amps, 0.0, paired=False, alternative="greater", label=f"H3_{key}")
        results[key] = {"result": result, "n": n, "mean": m, "sd": sd}
    return results


# ANALYSIS 4: SSA Control (deviant vs Block C) – confirmatory
def analysis_ssa(df: pd.DataFrame) -> dict:
    _print_section("ANALYSIS 4: SSA Control – deviant vs Block C (confirmatory)")
    results = {}
    for key, col in [("ssa_AC", "ssa_AC_amp_Fz"), ("ssa_BC", "ssa_BC_amp_Fz")]:
        amps = df[col].dropna()
        n = len(amps)
        m = float(amps.mean())
        sd = float(amps.std(ddof=1))
        log.info("\n[SSA] %s: N=%d, M=%.3f µV, SD=%.3f µV", key.upper(), n, m, sd)
        # alternative='less' because we expect deviant to be MORE NEGATIVE than control
        result = _safe_ttest(amps, 0.0, paired=False, alternative="less", label=f"SSA_{key}")
        results[key] = {"result": result, "n": n, "mean": m, "sd": sd}
        if result is not None:
            bf_val = result.get("BF10", np.nan)
            if not np.isnan(bf_val):
                log.info(_interpret_bf(bf_val, getattr(config, "BF10_THRESHOLD", 3.0)))
    return results


# ANALYSIS 5: Cluster permutation (sketch)
def analysis_cluster_permutation_from_csv(df: pd.DataFrame) -> None:
    _print_section("ANALYSIS 5: Cluster permutation – INSTRUCTIONS")
    log.info("Cluster test requires Evoked data (MNE). Call run_cluster_permutation() directly using Evoked lists.")


# ANALYSIS 6: Descriptive Statistics
def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    _print_section("ANALYSIS 6: Descriptive Statistics")
    conditions = {
        "MMN_A (Near-deviant, block A)": {"amp_col": "mmn_A_amp_Fz", "fal_col": "mmn_A_fal_ms", "n_col": "n_trials_dev_A"},
        "MMN_B (Far-deviant, block B)": {"amp_col": "mmn_B_amp_Fz", "fal_col": "mmn_B_fal_ms", "n_col": "n_trials_dev_B"},
        "P3a_A (deviant block A)": {"amp_col": "p3a_A_amp_Cz", "fal_col": "p3a_A_fal_ms", "n_col": "n_trials_dev_A"},
        "P3a_B (deviant block B)": {"amp_col": "p3a_B_amp_Cz", "fal_col": "p3a_B_fal_ms", "n_col": "n_trials_dev_B"},
        "SSA_AC (devA–ctrlC)": {"amp_col": "ssa_AC_amp_Fz", "fal_col": None, "n_col": "n_trials_dev_A"},
        "SSA_BC (devB–ctrlC)": {"amp_col": "ssa_BC_amp_Fz", "fal_col": None, "n_col": "n_trials_dev_B"},
    }
    rows = []
    for cond, cols in conditions.items():
        amp_s = df[cols["amp_col"]].dropna() if cols["amp_col"] in df else pd.Series(dtype=float)
        fal_s = df[cols["fal_col"]].dropna() if cols["fal_col"] and cols["fal_col"] in df else pd.Series(dtype=float)
        n_s = df[cols["n_col"]].dropna() if cols["n_col"] in df else pd.Series(dtype=float)
        row = {
            "Condition": cond,
            "N_participants": len(amp_s),
            "N_trials_mean": round(float(n_s.mean()), 1) if len(n_s) > 0 else np.nan,
            "N_trials_SD": round(float(n_s.std(ddof=1)), 1) if len(n_s) > 1 else np.nan,
            "Amplitude_mean_uV": round(float(amp_s.mean()), 4) if len(amp_s) > 0 else np.nan,
            "Amplitude_SD": round(float(amp_s.std(ddof=1)), 4) if len(amp_s) > 1 else np.nan,
            "FAL_mean_ms": round(float(fal_s.mean()), 2) if len(fal_s) > 0 else np.nan,
            "FAL_SD_ms": round(float(fal_s.std(ddof=1)), 2) if len(fal_s) > 1 else np.nan,
        }
        rows.append(row)
    desc_df = pd.DataFrame(rows)
    log.info("\n%s", desc_df.to_string(index=False))
    out = Path(config.STATS_PATH) / "descriptive_statistics.csv"
    desc_df.to_csv(out, index=False)
    log.info("Saved descriptive statistics: %s", out)
    return desc_df


# ANALYSIS 7: IEEE Formatting (updated to include SSA)
def generate_ieee_report(
    h1_results: dict,
    h2_results: dict,
    h3_results: dict,
    ssa_results: dict,
    df: pd.DataFrame,
) -> str:
    lines = [
        "=" * 70,
        "RESULTS IN IEEE FORMAT – ready for the Results section",
        "=" * 70,
        "",
        "HYPOTHESIS H1: MMN Significance (block A – Near-deviant; block B – Far-deviant)",
        "-" * 70,
    ]
    for key in ("mmn_A", "mmn_B"):
        r = h1_results.get(key, {})
        label = "MMN_A (Near-deviant, block A)" if key == "mmn_A" else "MMN_B (Far-deviant, block B)"
        lines.append(
            _format_ieee_ttest(
                r.get("result"),
                r.get("mean", np.nan),
                r.get("sd", np.nan),
                label,
                note="One-sided hypothesis (alternative='less'); "
                     f"BF10 threshold = {getattr(config, 'BF10_THRESHOLD', 3.0)}",
            )
        )
    lines += ["", "HYPOTHESIS H2: Looming asymmetry – MMN_A vs MMN_B comparison (DIRECTIONAL)", "-" * 70]
    r_amp = h2_results.get("amplitude", {})
    ma = r_amp.get("mean_A", np.nan)
    mb = r_amp.get("mean_B", np.nan)
    diff_m = ma - mb if not (np.isnan(ma) or np.isnan(mb)) else np.nan
    lines.append(
        _format_ieee_ttest(
            r_amp.get("result"),
            diff_m,
            r_amp.get("sd_A", np.nan),
            "H2 – MMN_A vs MMN_B Amplitude (Fz)",
            note=f"MMN_A: M = {ma:.3f} µV, SD = {r_amp.get('sd_A', np.nan):.3f}; "
                 f"MMN_B: M = {mb:.3f} µV, SD = {r_amp.get('sd_B', np.nan):.3f}. "
                 f"DIRECTIONAL test: A < B (more negative for looming).",
        )
    )
    r_fal = h2_results.get("fal", {})
    fa = r_fal.get("mean_A", np.nan)
    fb = r_fal.get("mean_B", np.nan)
    diff_fal = fa - fb if not (np.isnan(fa) or np.isnan(fb)) else np.nan
    lines.append(
        _format_ieee_ttest(
            r_fal.get("result"),
            diff_fal,
            r_fal.get("sd_A", np.nan),
            "H2 – MMN_A vs MMN_B FAL (Fz)",
            note=f"FAL_A: M = {fa:.1f} ms, SD = {r_fal.get('sd_A', np.nan):.1f}; "
                 f"FAL_B: M = {fb:.1f} ms, SD = {r_fal.get('sd_B', np.nan):.1f}. "
                 f"DIRECTIONAL test: A < B (shorter latency for looming).",
            unit="ms",
        )
    )
    lines += ["", "HYPOTHESIS H3: P3a Significance", "-" * 70]
    for key in ("p3a_A", "p3a_B"):
        r = h3_results.get(key, {})
        label = "P3a_A (deviant block A)" if key == "p3a_A" else "P3a_B (deviant block B)"
        lines.append(
            _format_ieee_ttest(
                r.get("result"),
                r.get("mean", np.nan),
                r.get("sd", np.nan),
                label,
                note="One-sided hypothesis (alternative='greater', expected positive P3a amplitude)",
            )
        )
    # --- SSA Control ---
    lines += ["", "SSA CONTROL (deviant vs Block C) – confirmatory test", "-" * 70]
    for key in ("ssa_AC", "ssa_BC"):
        r = ssa_results.get(key, {})
        label = "SSA_AC (dev_A – ctrl_C)" if key == "ssa_AC" else "SSA_BC (dev_B – ctrl_C)"
        lines.append(
            _format_ieee_ttest(
                r.get("result"),
                r.get("mean", np.nan),
                r.get("sd", np.nan),
                label,
                note="One-sided hypothesis (alternative='less'): expected negative amplitude (deviant more negative than control). "
                     f"BF10 threshold = {getattr(config, 'BF10_THRESHOLD', 3.0)}",
            )
        )
    lines += [
        "",
        "=" * 70,
        "INTERPRETATION NOTES:",
        f"  • BF10 > {getattr(config, 'BF10_THRESHOLD', 3.0)} → moderate evidence.",
        "  • Bayesian tests use a Cauchy prior with scale r = 0.5, as pre-registered.",
        "  • MMN results for Fz electrode, P3a for Cz (FCz unavailable in montage).",
        "=" * 70,
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    log.info("=" * 60)
    log.info("03_statistics.py – START")
    log.info("=" * 60)

    Path(config.STATS_PATH).mkdir(parents=True, exist_ok=True)
    df = load_erp_measures()

    h1_results = analysis_h1(df)
    h2_results = analysis_h2(df)
    h3_results = analysis_h3(df)
    ssa_results = analysis_ssa(df)

    analysis_cluster_permutation_from_csv(df)
    descriptive_statistics(df)

    _print_section("ANALYSIS 7: Generating IEEE report")
    ieee_text = generate_ieee_report(h1_results, h2_results, h3_results, ssa_results, df)
    ieee_path = Path(config.STATS_PATH) / "results_ieee_format.txt"
    with open(ieee_path, "w", encoding="utf-8") as f:
        f.write(ieee_text)
    log.info("Saved IEEE report: %s", ieee_path)
    print("\n" + ieee_text)

    log.info("=" * 60)
    log.info("03_statistics.py – FINISHED SUCCESSFULLY")
    log.info("=" * 60)


if __name__ == "__main__":
    main()