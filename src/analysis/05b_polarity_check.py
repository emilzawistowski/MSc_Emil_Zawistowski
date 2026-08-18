import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import mne

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
mne.set_log_level("ERROR")

OUT_CSV = cfg.STATS_PATH / "mastoid_polarity_check.csv"
OUT_TXT = cfg.REPORTS_PATH / "mastoid_polarity_check.txt"

MASTOID_CHANNELS = ["TP9", "TP10"]
FRONTAL_CHANNEL = "Fz"


def load_epochs(participant: str, block: str, condition: str) -> mne.Epochs | None:
    if cfg.is_excluded(participant, block):
        return None
    fname = Path(cfg.PREPROCESSED_PATH) / f"{participant}_block{block}_{condition}_epo.fif"
    try:
        return mne.read_epochs(fname, preload=True, verbose=False)
    except FileNotFoundError:
        return None
    except Exception as exc:
        log.error("Error loading %s: %s", fname.name, exc)
        return None


def has_mastoid_channels(evoked: mne.Evoked) -> list[str]:
    return [ch for ch in MASTOID_CHANNELS if ch not in evoked.ch_names]


def window_mean_amplitude(evoked: mne.Evoked, ch_name: str, tmin: float, tmax: float) -> float:
    times = evoked.times
    mask = (times >= tmin) & (times <= tmax)
    if not np.any(mask) or ch_name not in evoked.ch_names:
        return np.nan
    idx = evoked.ch_names.index(ch_name)
    return float(np.mean(evoked.data[idx, mask]) * 1e6)


def main() -> None:
    log.info("=" * 60)
    log.info("05b_mastoid_polarity_check.py -- START")
    log.info("=" * 60)

    cfg.STATS_PATH.mkdir(parents=True, exist_ok=True)
    cfg.REPORTS_PATH.mkdir(parents=True, exist_ok=True)

    rows = []
    car_mmn_A_evokeds = []
    car_mmn_B_evokeds = []

    for p in cfg.PARTICIPANTS:
        for block, mmn_store in (("A", car_mmn_A_evokeds), ("B", car_mmn_B_evokeds)):
            std_cond, dev_cond = "std", "dev"
            epochs_std = load_epochs(p, block, std_cond)
            epochs_dev = load_epochs(p, block, dev_cond)
            if epochs_std is None or epochs_dev is None:
                continue

            try:
                evk_std = epochs_std.average()
                evk_dev = epochs_dev.average()
                mmn_car = mne.combine_evoked([evk_dev, evk_std], weights=[1, -1])
                mmn_car.comment = f"{p}_mmn_{block}"
            except Exception as exc:                
                log.error("combine_evoked failed for %s block %s: %s", p, block, exc)
                continue

            missing = has_mastoid_channels(mmn_car)
            fz_car = window_mean_amplitude(mmn_car, FRONTAL_CHANNEL, cfg.MMN_TMIN, cfg.MMN_TMAX)

            if missing:
                rows.append({
                    "participant": p, "block": block,
                    "fz_amp_car_uV": round(fz_car, 4) if not np.isnan(fz_car) else np.nan,
                    "tp9_amp_car_uV": np.nan,
                    "tp10_amp_car_uV": np.nan,
                    "reversal_tp9": np.nan,
                    "reversal_tp10": np.nan,
                    "reversal_either": np.nan,
                    "reversal_both": np.nan,
                    "note": f"missing channel(s): {', '.join(missing)}",
                })
                continue

            tp9_car = window_mean_amplitude(mmn_car, "TP9", cfg.MMN_TMIN, cfg.MMN_TMAX)
            tp10_car = window_mean_amplitude(mmn_car, "TP10", cfg.MMN_TMIN, cfg.MMN_TMAX)

            if np.isnan(fz_car) or np.isnan(tp9_car) or np.isnan(tp10_car):
                rev_tp9 = rev_tp10 = rev_either = rev_both = np.nan
            else:
                rev_tp9 = bool(fz_car < 0 and tp9_car > 0)
                rev_tp10 = bool(fz_car < 0 and tp10_car > 0)
                rev_either = bool(rev_tp9 or rev_tp10)
                rev_both = bool(rev_tp9 and rev_tp10)

            rows.append({
                "participant": p, "block": block,
                "fz_amp_car_uV": round(fz_car, 4) if not np.isnan(fz_car) else np.nan,
                "tp9_amp_car_uV": round(tp9_car, 4) if not np.isnan(tp9_car) else np.nan,
                "tp10_amp_car_uV": round(tp10_car, 4) if not np.isnan(tp10_car) else np.nan,
                "reversal_tp9": rev_tp9,
                "reversal_tp10": rev_tp10,
                "reversal_either": rev_either,
                "reversal_both": rev_both,
                "note": "",
            })
            mmn_store.append(mmn_car)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    log.info("Saved: %s", OUT_CSV)

    def _ga_summary(evokeds: list[mne.Evoked], label: str) -> str:
        if not evokeds:
            return f"{label}: no participants with usable TP9/TP10 - grand average not computed.\n"
        ga = mne.grand_average(evokeds)
        fz = window_mean_amplitude(ga, FRONTAL_CHANNEL, cfg.MMN_TMIN, cfg.MMN_TMAX)
        tp9 = window_mean_amplitude(ga, "TP9", cfg.MMN_TMIN, cfg.MMN_TMAX)
        tp10 = window_mean_amplitude(ga, "TP10", cfg.MMN_TMIN, cfg.MMN_TMAX)
        rev_tp9 = fz < 0 and tp9 > 0
        rev_tp10 = fz < 0 and tp10 > 0
        return (
            f"{label} (N={len(evokeds)}, grand average, CAR reference, "
            f"window {cfg.MMN_TMIN*1e3:.0f}-{cfg.MMN_TMAX*1e3:.0f} ms):\n"
            f"    Fz   = {fz:+.3f} uV\n"
            f"    TP9  = {tp9:+.3f} uV   (reversal vs Fz: {'PASS' if rev_tp9 else 'NOT OBSERVED'})\n"
            f"    TP10 = {tp10:+.3f} uV   (reversal vs Fz: {'PASS' if rev_tp10 else 'NOT OBSERVED'})\n"
        )

    n_either_A = df.loc[df["block"] == "A", "reversal_either"].sum()
    n_both_A = df.loc[df["block"] == "A", "reversal_both"].sum()
    n_valid_A = df.loc[df["block"] == "A", "reversal_either"].notna().sum()
    n_either_B = df.loc[df["block"] == "B", "reversal_either"].sum()
    n_both_B = df.loc[df["block"] == "B", "reversal_both"].sum()
    n_valid_B = df.loc[df["block"] == "B", "reversal_either"].notna().sum()

    report_lines = [
        "=" * 70,
        "MASTOID POLARITY-REVERSAL CHECK (CAR reference, sign comparison)",
        "=" * 70,
        "",
        "This check compares the sign of Fz against TP9 and TP10 independently,",
        "on the CAR reference used throughout the main pipeline (not re-referenced",
        "to TP9/TP10, which would force them to be equal-and-opposite by",
        "construction). TP9/TP10 approximate the true mastoids per the CLACS-32",
        "montage. Interpret as a supportive, secondary diagnostic - see thesis",
        "Limitations.",
        "",
        _ga_summary(car_mmn_A_evokeds, "Block A (Near-deviant)"),
        _ga_summary(car_mmn_B_evokeds, "Block B (Far-deviant)"),
        "-" * 70,
        "PER-PARTICIPANT SUMMARY:",
        f"  Block A: reversal at TP9 and/or TP10 in {int(n_either_A)} / {int(n_valid_A)} participants; "
        f"at BOTH in {int(n_both_A)} / {int(n_valid_A)}.",
        f"  Block B: reversal at TP9 and/or TP10 in {int(n_either_B)} / {int(n_valid_B)} participants; "
        f"at BOTH in {int(n_both_B)} / {int(n_valid_B)}.",
        "",
        f"Full per-participant table: {OUT_CSV}",
        "=" * 70,
    ]
    report = "\n".join(report_lines)

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(report)
    log.info("Saved: %s", OUT_TXT)


    log.info("=" * 60)
    log.info("05b_mastoid_polarity_check.py -- FINISHED SUCCESSFULLY")
    log.info("=" * 60)


if __name__ == "__main__":
    main()