import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mne
import numpy as np
import pandas as pd
import sys
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

mne.set_log_level("WARNING")


COLORS = {
    "std_A":   "#0072B2",         
    "dev_A":   "#D55E00",               
    "std_B":   "#009E73",          
    "dev_B":   "#CC79A7",         
    "ctrl_C":  "#E69F00",           
    "mmn_A":   "#D55E00",
    "mmn_B":   "#0072B2",
    "ssa_AC":  "#D55E00",
    "ssa_BC":  "#0072B2",
}

MMN_COLOR   = "#D55E00"
P3A_COLOR   = "#009E73"
ALPHA_SHADE = 0.20


def load_epochs(participant: str, block: str, condition: str) -> mne.Epochs | None:
    """Loads a .fif epochs file for a given participant/block/condition."""
    if config.is_excluded(participant, block):
        log.warning("Excluded: %s block %s (see excluded_recordings.csv) - skipping.",
                    participant, block)
        return None

    fname = Path(config.PREPROCESSED_PATH) / f"{participant}_block{block}_{condition}_epo.fif"
    try:
        epochs = mne.read_epochs(fname, preload=True, verbose=False)
        log.info("Loaded: %s  (%d epochs)", fname.name, len(epochs))
        return epochs
    except FileNotFoundError:
        log.warning("File missing: %s – skipping participant.", fname.name)
        return None
    except Exception as exc:                
        log.error("Error while loading %s: %s", fname.name, exc)
        return None


def safe_average(epochs: mne.Epochs | None, label: str = "") -> mne.Evoked | None:
    """Returns ERP (.average()) or None if epochs is None."""
    if epochs is None:
        return None
    try:
        evoked = epochs.average()
        evoked.comment = label
        return evoked
    except Exception as exc:                
        log.error("Error in .average() for '%s': %s", label, exc)
        return None


def extract_mean_amplitude(
    evoked: mne.Evoked,
    tmin: float,
    tmax: float,
    picks: list[str],
) -> float:
    """Mean amplitude (uV) in [tmin, tmax], averaged over the given channels."""
    try:
        times = evoked.times
        mask = (times >= tmin) & (times <= tmax)
        if not np.any(mask):
            log.warning("Window %.0f–%.0f ms outside time range.", tmin * 1e3, tmax * 1e3)
            return np.nan

        ch_indices = mne.pick_channels(evoked.ch_names, include=picks)
        if len(ch_indices) == 0:
            log.warning("None of the channels %s found.", picks)
            return np.nan

        data = evoked.data[ch_indices][:, mask]                              
        return float(np.mean(data) * 1e6)                  
    except Exception as exc:                
        log.error("extract_mean_amplitude error: %s", exc)
        return np.nan


def compute_fal(
    evoked: mne.Evoked,
    tmin: float,
    tmax: float,
    ch_name: str,
    fraction: float = 0.5,
    polarity: str = "negative",
) -> float:
    """Fractional Area Latency: the time at which the cumulative area under
    the (rectified) curve reaches `fraction` of the total area. For
    polarity='negative' (MMN) the signal is inverted before rectifying;
    for 'positive' (P3a) it is used as-is."""
    try:
        times = evoked.times
        mask  = (times >= tmin) & (times <= tmax)
        if not np.any(mask):
            return np.nan

        ch_idx = mne.pick_channels(evoked.ch_names, include=[ch_name])
        if len(ch_idx) == 0:
            log.warning("Electrode '%s' not found.", ch_name)
            return np.nan

        signal = evoked.data[ch_idx[0], mask].copy() * 1e6

        if polarity == "negative":
            signal = -signal
            signal = np.clip(signal, 0, None)
        else:
            signal = np.clip(signal, 0, None)

        total_area = np.trapezoid(signal, times[mask])
        if total_area <= 0:
            log.warning("FAL: area under curve = 0 or negative (%s, %s).", ch_name, polarity)
            return np.nan

        cumulative = np.array([
            np.trapezoid(signal[:i + 1], times[mask][:i + 1])
            for i in range(len(signal))
        ])
        idx = np.searchsorted(cumulative, fraction * total_area)
        idx = min(idx, len(times[mask]) - 1)
        return float(times[mask][idx] * 1e3)           
    except Exception as exc:                
        log.error("compute_fal error (%s): %s", ch_name, exc)
        return np.nan


MIN_PARTICIPANTS = 5


def safe_grand_average(
    evoked_list: list[mne.Evoked | None],
    label: str = "",
) -> mne.Evoked | None:
    """Grand average with a minimum-participant-count check."""
    valid = [e for e in evoked_list if e is not None]
    n = len(valid)
    if n < MIN_PARTICIPANTS:
        log.warning(
            "Grand average '%s': only %d/%d participants – skipping (requires ≥ %d).",
            label, n, len(evoked_list), MIN_PARTICIPANTS,
        )
        return None
    log.info("Grand average '%s': N=%d participants.", label, n)
    ga = mne.grand_average(valid)
    ga.comment = label
    return ga


def _mmn_p3a_rectangles(ax: plt.Axes) -> None:
    """Adds semi-transparent rectangles marking MMN and P3a windows."""
    mmn_rect = mpatches.FancyArrowPatch
    ax.axvspan(
        config.MMN_TMIN * 1e3, config.MMN_TMAX * 1e3,
        alpha=0.12, color=MMN_COLOR, label="MMN window",
    )
    ax.axvspan(
        config.P3A_TMIN * 1e3, config.P3A_TMAX * 1e3,
        alpha=0.12, color=P3A_COLOR, label="P3a window",
    )


def _sem_shade(
    ax: plt.Axes,
    times_ms: np.ndarray,
    evoked_list: list[mne.Evoked],
    ch_name: str,
    color: str,
) -> None:
    """Draws SEM shading around the grand average."""
    try:
        ch_idx = mne.pick_channels(evoked_list[0].ch_names, include=[ch_name])
        if not ch_idx:
            return
        signals = np.array([e.data[ch_idx[0]] * 1e6 for e in evoked_list])
        sem = signals.std(axis=0, ddof=1) / np.sqrt(len(signals))
        mean = signals.mean(axis=0)
        ax.fill_between(times_ms, mean - sem, mean + sem, alpha=ALPHA_SHADE, color=color)
    except Exception as exc:                
        log.warning("_sem_shade error: %s", exc)


def plot_grand_avg_all_conditions(
    ga_dict: dict[str, mne.Evoked | None],
    sem_dict: dict[str, list[mne.Evoked]],
    ch_name: str = "Fz",
) -> None:
    """Figure a): superimposed 5-condition grand average waveforms with SEM on channel Fz."""
    fig, ax = plt.subplots(figsize=(10, 5))
    times_ms = list(ga_dict.values())[0].times * 1e3

    for key, ga in ga_dict.items():
        if ga is None:
            continue
        ch_idx = mne.pick_channels(ga.ch_names, include=[ch_name])
        if not ch_idx:
            continue
        signal = ga.data[ch_idx[0]] * 1e6
        color  = COLORS.get(key, "black")
        ax.plot(times_ms, signal, color=color, lw=1.8, label=key)
        if key in sem_dict and sem_dict[key]:
            _sem_shade(ax, times_ms, sem_dict[key], ch_name, color)

    _mmn_p3a_rectangles(ax)
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.axvline(0, color="k", lw=0.6, ls="--")
    ax.set_xlim(times_ms[0], times_ms[-1])
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Amplitude [µV]")
    ax.legend(fontsize=8, ncol=3)
    ax.invert_yaxis()
    fig.tight_layout()

    out = Path(config.FIGURES_ERP_PATH) / "grand_avg_all_conditions.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    log.info("Saved: %s", out)


def plot_mmn_difference_waves(
    ga_mmn_A: mne.Evoked | None,
    ga_mmn_B: mne.Evoked | None,
    sem_mmnA: list[mne.Evoked],
    sem_mmnB: list[mne.Evoked],
    ch_name: str = "Fz",
) -> None:
    """Figure b): MMN_A vs MMN_B difference waves with SEM on Fz."""
    fig, ax = plt.subplots(figsize=(10, 4))

    _ref_ga = next((g for g in (ga_mmn_A, ga_mmn_B) if g is not None), None)
    times_ms = _ref_ga.times * 1e3 if _ref_ga is not None else None

    for ga, sem_list, key in [
        (ga_mmn_A, sem_mmnA, "mmn_A"),
        (ga_mmn_B, sem_mmnB, "mmn_B"),
    ]:
        if ga is None:
            continue
        ch_idx = mne.pick_channels(ga.ch_names, include=[ch_name])
        if not ch_idx:
            continue
        this_times_ms = ga.times * 1e3
        signal   = ga.data[ch_idx[0]] * 1e6
        color    = COLORS[key]
        ax.plot(this_times_ms, signal, color=color, lw=1.8, label=key)
        if sem_list:
            _sem_shade(ax, this_times_ms, sem_list, ch_name, color)

    _mmn_p3a_rectangles(ax)
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.axvline(0, color="k", lw=0.6, ls="--")
    if times_ms is not None:
        ax.set_xlim(times_ms[0], times_ms[-1])
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Amplitude [µV]")
    ax.legend()
    ax.invert_yaxis()
    fig.tight_layout()

    out = Path(config.FIGURES_ERP_PATH) / "mmn_difference_waves.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    log.info("Saved: %s", out)


def _sphere_covering_all_channels(info: mne.Info, margin: float = 1.03) -> tuple:
    """Explicit (x, y, z, radius) sphere for plot_topomap, sized to cover
    the most eccentric channel, unlike MNE's default sphere='auto' fit."""
    pos = np.array([ch['loc'][:3] for ch in info['chs'] if ch['kind'] == 2])                
    center_xy = pos[:, :2].mean(axis=0)
    radii = np.sqrt(((pos[:, :2] - center_xy) ** 2).sum(axis=1))
    r = float(radii.max()) * margin
    return (float(center_xy[0]), float(center_xy[1]), 0.0, r)


def plot_topomap_mmn(
    ga: mne.Evoked | None,
    block: str,
    t_center_s: float | None = None,
) -> None:
    """Figure c): topographical map for MMN_A or MMN_B."""
    if ga is None:
        log.warning("Missing grand average MMN_%s – skipping topomap.", block)
        return
    t = t_center_s if t_center_s is not None else (config.MMN_TMIN + config.MMN_TMAX) / 2

    try:
        sphere = _sphere_covering_all_channels(ga.info)

        fig = ga.plot_topomap(
            times=[t],
            average=(config.MMN_TMAX - config.MMN_TMIN),
            sphere=sphere,
            extrapolate='head',
            show=False,
        )
        fig.set_size_inches(5, 5.4)

        topo_ax = fig.axes[0]
        topo_ax.set_title(topo_ax.get_title(), fontsize=9, color='dimgray', pad=4)

        out = Path(config.FIGURES_TOPO_PATH) / f"topomap_mmn_{block}.png"
        fig.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.15)
        plt.close(fig)
        log.info("Saved: %s", out)
    except Exception as exc:                
        log.error("plot_topomap_mmn_%s error: %s", block, exc)


def _hstack_pngs_with_labels(png_paths: list[Path], labels: list[str], out_path: Path,
                              gap_px: int = 30, label_fontsize: int = 22) -> None:
    """Composites already-saved single-panel PNGs side by side via PIL."""
    imgs = [Image.open(p) for p in png_paths]
    label_h = int(label_fontsize * 1.8)
    max_h = max(im.height for im in imgs)
    total_w = sum(im.width for im in imgs) + gap_px * (len(imgs) - 1)
    canvas = Image.new("RGB", (total_w, max_h + label_h), "white")

    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", label_fontsize)
    except OSError:
        font = ImageFont.load_default()

    x = 0
    for im, label in zip(imgs, labels):
        canvas.paste(im, (x, 0))
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        draw.text((x + im.width / 2 - text_w / 2, max_h + 4), label,
                   fill="black", font=font)
        x += im.width + gap_px

    canvas.save(out_path)
    for im in imgs:
        im.close()


def plot_topomap_mmn_combined(
    ga_mmn_A: mne.Evoked | None,
    ga_mmn_B: mne.Evoked | None,
) -> None:
    """Block A and Block B MMN topomaps side by side, composited from two
    single-panel renders via _hstack_pngs_with_labels()."""
    evokeds = [("A", ga_mmn_A), ("B", ga_mmn_B)]
    present = [(k, v) for k, v in evokeds if v is not None]
    if not present:
        log.warning("No data for combined MMN topomap.")
        return

    t = (config.MMN_TMIN + config.MMN_TMAX) / 2
    tmp_paths, labels = [], []

    for key, ga in present:
        try:
            sphere = _sphere_covering_all_channels(ga.info)
            fig = ga.plot_topomap(
                times=[t],
                average=(config.MMN_TMAX - config.MMN_TMIN),
                sphere=sphere,
                extrapolate='head',
                show=False,
            )
            fig.set_size_inches(5, 5.4)
            topo_ax = fig.axes[0]
            topo_ax.set_title(topo_ax.get_title(), fontsize=9, color='dimgray', pad=4)

            tmp_out = Path(config.FIGURES_TOPO_PATH) / f"_tmp_topomap_mmn_{key}.png"
            fig.savefig(tmp_out, dpi=300, bbox_inches='tight', pad_inches=0.15)
            plt.close(fig)
            tmp_paths.append(tmp_out)
            labels.append(f"Block {key}")
        except Exception as exc:                
            log.error("plot_topomap_mmn_combined block %s: %s", key, exc)

    if not tmp_paths:
        log.warning("No panels rendered for combined MMN topomap.")
        return

    out = Path(config.FIGURES_TOPO_PATH) / "topomap_mmn.png"
    _hstack_pngs_with_labels(tmp_paths, labels, out)
    for p in tmp_paths:
        p.unlink(missing_ok=True)
    log.info("Saved: %s", out)


def plot_topomap_p3a(
    ga_p3a_A: mne.Evoked | None,
    ga_p3a_B: mne.Evoked | None,
) -> None:
    """Figure d): P3a topomaps on the deviant-minus-standard difference wave."""
    t = (config.P3A_TMIN + config.P3A_TMAX) / 2
    evokeds = {k: v for k, v in [("A", ga_p3a_A), ("B", ga_p3a_B)] if v is not None}
    if not evokeds:
        log.warning("No data for P3a topomaps.")
        return

    for key, ga in evokeds.items():
        try:
            sphere = _sphere_covering_all_channels(ga.info)

            fig = ga.plot_topomap(
                times=[t],
                average=(config.P3A_TMAX - config.P3A_TMIN),
                sphere=sphere,
                extrapolate='head',
                show=False,
            )
            fig.set_size_inches(5, 5.4)

            topo_ax = fig.axes[0]
            topo_ax.set_title(topo_ax.get_title(), fontsize=9, color='dimgray', pad=4)

            out = Path(config.FIGURES_TOPO_PATH) / f"topomap_p3a_{key}.png"
            fig.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.15)
            plt.close(fig)
            log.info("Saved: %s", out)
        except Exception as exc:                
            log.error("plot_topomap_p3a_%s error: %s", key, exc)


def plot_topomap_p3a_combined(
    ga_p3a_A: mne.Evoked | None,
    ga_p3a_B: mne.Evoked | None,
) -> None:
    """Block A and Block B P3a topomaps side by side, composited from two
    single-panel renders via _hstack_pngs_with_labels()."""
    evokeds = [("A", ga_p3a_A), ("B", ga_p3a_B)]
    present = [(k, v) for k, v in evokeds if v is not None]
    if not present:
        log.warning("No data for combined P3a topomap.")
        return

    t = (config.P3A_TMIN + config.P3A_TMAX) / 2
    tmp_paths, labels = [], []

    for key, ga in present:
        try:
            sphere = _sphere_covering_all_channels(ga.info)
            fig = ga.plot_topomap(
                times=[t],
                average=(config.P3A_TMAX - config.P3A_TMIN),
                sphere=sphere,
                extrapolate='head',
                show=False,
            )
            fig.set_size_inches(5, 5.4)
            topo_ax = fig.axes[0]
            topo_ax.set_title(topo_ax.get_title(), fontsize=9, color='dimgray', pad=4)

            tmp_out = Path(config.FIGURES_TOPO_PATH) / f"_tmp_topomap_p3a_{key}.png"
            fig.savefig(tmp_out, dpi=300, bbox_inches='tight', pad_inches=0.15)
            plt.close(fig)
            tmp_paths.append(tmp_out)
            labels.append(f"Block {key}")
        except Exception as exc:                
            log.error("plot_topomap_p3a_combined block %s: %s", key, exc)

    if not tmp_paths:
        log.warning("No panels rendered for combined P3a topomap.")
        return

    out = Path(config.FIGURES_TOPO_PATH) / "topomap_p3a.png"
    _hstack_pngs_with_labels(tmp_paths, labels, out)
    for p in tmp_paths:
        p.unlink(missing_ok=True)
    log.info("Saved: %s", out)


def plot_ssa_control(
    ga_ssa_AC: mne.Evoked | None,
    ga_ssa_BC: mne.Evoked | None,
    sem_AC: list[mne.Evoked],
    sem_BC: list[mne.Evoked],
    ch_name: str = "Fz",
) -> None:
    """Figure e): SSA control – diff_A_vs_C and diff_B_vs_C."""
    fig, ax = plt.subplots(figsize=(10, 4))


    _ref_ga = next((g for g in (ga_ssa_AC, ga_ssa_BC) if g is not None), None)
    times_ms = _ref_ga.times * 1e3 if _ref_ga is not None else None

    for ga, sem_list, key in [
        (ga_ssa_AC, sem_AC, "ssa_AC"),
        (ga_ssa_BC, sem_BC, "ssa_BC"),
    ]:
        if ga is None:
            continue
        ch_idx = mne.pick_channels(ga.ch_names, include=[ch_name])
        if not ch_idx:
            continue
        this_times_ms = ga.times * 1e3
        signal   = ga.data[ch_idx[0]] * 1e6
        color    = COLORS[key]
        label    = "dev_A – ctrl_C" if "AC" in key else "dev_B – ctrl_C"
        ax.plot(this_times_ms, signal, color=color, lw=1.8, label=label)
        if sem_list:
            _sem_shade(ax, this_times_ms, sem_list, ch_name, color)

    _mmn_p3a_rectangles(ax)
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.axvline(0, color="k", lw=0.6, ls="--")
    if times_ms is not None:
        ax.set_xlim(times_ms[0], times_ms[-1])
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Amplitude [µV]")
    ax.legend()
    ax.invert_yaxis()
    fig.tight_layout()

    out = Path(config.FIGURES_ERP_PATH) / "ssa_control.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    log.info("Saved: %s", out)


def plot_individual_erps(
    individual_data: dict[str, dict[str, mne.Evoked | None]],
    ch_name: str = "Fz",
) -> None:
    """Figure f): individual ERPs on Fz for each participant."""
    out_dir = Path(config.FIGURES_ERP_PATH) / "individual_erps"
    out_dir.mkdir(parents=True, exist_ok=True)

    condition_keys = ["std_A", "dev_A", "std_B", "dev_B", "ctrl_C"]

    for pid, cond_dict in individual_data.items():
        fig, ax = plt.subplots(figsize=(10, 4))
        has_data = False

        for key in condition_keys:
            evoked = cond_dict.get(key)
            if evoked is None:
                continue
            ch_idx = mne.pick_channels(evoked.ch_names, include=[ch_name])
            if not ch_idx:
                continue
            times_ms = evoked.times * 1e3
            signal   = evoked.data[ch_idx[0]] * 1e6
            ax.plot(times_ms, signal, color=COLORS.get(key, "gray"), lw=1.4, label=key)
            has_data = True

        if not has_data:
            plt.close(fig)
            continue

        _mmn_p3a_rectangles(ax)
        ax.axhline(0, color="k", lw=0.6, ls="--")
        ax.axvline(0, color="k", lw=0.6, ls="--")
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Amplitude [µV]")


        ax.legend(fontsize=8)
        ax.invert_yaxis()
        fig.tight_layout()

        out = out_dir / f"erp_{pid}.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)

    log.info("Saved individual plots in: %s", out_dir)


def main() -> None:
    log.info("=" * 60)
    log.info("02_erp_analysis.py – START")
    log.info("=" * 60)

    for d in [config.FIGURES_ERP_PATH, config.FIGURES_TOPO_PATH, config.STATS_PATH]:
        Path(d).mkdir(parents=True, exist_ok=True)

    participants = config.PARTICIPANTS

    erp: dict[str, dict[str, mne.Evoked | None]] = {p: {} for p in participants}

    for p in participants:
        log.info("--- Participant: %s ---", p)


        epochs_std_A = load_epochs(p, "A", "std")
        epochs_dev_A = load_epochs(p, "A", "dev")
        erp[p]["std_A"] = safe_average(epochs_std_A, f"{p}_std_A")
        erp[p]["dev_A"] = safe_average(epochs_dev_A, f"{p}_dev_A")


        epochs_std_B = load_epochs(p, "B", "std")
        epochs_dev_B = load_epochs(p, "B", "dev")
        erp[p]["std_B"] = safe_average(epochs_std_B, f"{p}_std_B")
        erp[p]["dev_B"] = safe_average(epochs_dev_B, f"{p}_dev_B")


        epochs_ctrl_C = load_epochs(p, "C", "std")
        erp[p]["ctrl_C"] = safe_average(epochs_ctrl_C, f"{p}_ctrl_C")

        erp[p]["_n_std_A"] = len(epochs_std_A) if epochs_std_A is not None else 0
        erp[p]["_n_dev_A"] = len(epochs_dev_A) if epochs_dev_A is not None else 0
        erp[p]["_n_std_B"] = len(epochs_std_B) if epochs_std_B is not None else 0
        erp[p]["_n_dev_B"] = len(epochs_dev_B) if epochs_dev_B is not None else 0
        erp[p]["_n_ctrl_C"] = len(epochs_ctrl_C) if epochs_ctrl_C is not None else 0

    mmn_per_p:     dict[str, mne.Evoked | None] = {}
    mmn_B_per_p:   dict[str, mne.Evoked | None] = {}
    ssa_AC_per_p:  dict[str, mne.Evoked | None] = {}
    ssa_BC_per_p:  dict[str, mne.Evoked | None] = {}

    for p in participants:

        if erp[p]["dev_A"] is not None and erp[p]["std_A"] is not None:
            try:
                mmn_per_p[p] = mne.combine_evoked(
                    [erp[p]["dev_A"], erp[p]["std_A"]], weights=[1, -1]
                )
                mmn_per_p[p].comment = f"{p}_mmn_A"
            except Exception as exc:                
                log.error("MMN_A combine_evoked %s: %s", p, exc)
                mmn_per_p[p] = None
        else:
            mmn_per_p[p] = None


        if erp[p]["dev_B"] is not None and erp[p]["std_B"] is not None:
            try:
                mmn_B_per_p[p] = mne.combine_evoked(
                    [erp[p]["dev_B"], erp[p]["std_B"]], weights=[1, -1]
                )
                mmn_B_per_p[p].comment = f"{p}_mmn_B"
            except Exception as exc:                
                log.error("MMN_B combine_evoked %s: %s", p, exc)
                mmn_B_per_p[p] = None
        else:
            mmn_B_per_p[p] = None


        if erp[p]["dev_A"] is not None and erp[p]["ctrl_C"] is not None:
            try:
                ssa_AC_per_p[p] = mne.combine_evoked(
                    [erp[p]["dev_A"], erp[p]["ctrl_C"]], weights=[1, -1]
                )
                ssa_AC_per_p[p].comment = f"{p}_ssa_AC"
            except Exception as exc:                
                log.error("SSA_AC combine_evoked %s: %s", p, exc)
                ssa_AC_per_p[p] = None
        else:
            ssa_AC_per_p[p] = None


        if erp[p]["dev_B"] is not None and erp[p]["ctrl_C"] is not None:
            try:
                ssa_BC_per_p[p] = mne.combine_evoked(
                    [erp[p]["dev_B"], erp[p]["ctrl_C"]], weights=[1, -1]
                )
                ssa_BC_per_p[p].comment = f"{p}_ssa_BC"
            except Exception as exc:                
                log.error("SSA_BC combine_evoked %s: %s", p, exc)
                ssa_BC_per_p[p] = None
        else:
            ssa_BC_per_p[p] = None

    def _list(d: dict) -> list:
        return [v for v in d.values() if v is not None]

    ga_std_A  = safe_grand_average([erp[p]["std_A"]  for p in participants], "GA_std_A")
    ga_dev_A  = safe_grand_average([erp[p]["dev_A"]  for p in participants], "GA_dev_A")
    ga_std_B  = safe_grand_average([erp[p]["std_B"]  for p in participants], "GA_std_B")
    ga_dev_B  = safe_grand_average([erp[p]["dev_B"]  for p in participants], "GA_dev_B")
    ga_ctrl_C = safe_grand_average([erp[p]["ctrl_C"] for p in participants], "GA_ctrl_C")

    ga_mmn_A  = safe_grand_average(list(mmn_per_p.values()),   "GA_mmn_A")
    ga_mmn_B  = safe_grand_average(list(mmn_B_per_p.values()), "GA_mmn_B")
    ga_ssa_AC = safe_grand_average(list(ssa_AC_per_p.values()), "GA_ssa_AC")
    ga_ssa_BC = safe_grand_average(list(ssa_BC_per_p.values()), "GA_ssa_BC")

    records = []
    fz_ch  = "Fz"
    cz_ch  = config.ROI_P3A[0] if config.ROI_P3A else "Cz"

    for p in participants:
        mmn_A_e  = mmn_per_p.get(p)
        mmn_B_e  = mmn_B_per_p.get(p)
        ssa_AC_e = ssa_AC_per_p.get(p)
        ssa_BC_e = ssa_BC_per_p.get(p)

        def _amp(evk, tmin, tmax, picks):
            return extract_mean_amplitude(evk, tmin, tmax, picks) if evk else np.nan

        def _fal(evk, tmin, tmax, ch, pol):
            return compute_fal(evk, tmin, tmax, ch, polarity=pol) if evk else np.nan

        row = {
            "participant":        p,

            "mmn_A_amp_Fz":       _amp(mmn_A_e,  config.MMN_TMIN, config.MMN_TMAX, [fz_ch]),
            "mmn_B_amp_Fz":       _amp(mmn_B_e,  config.MMN_TMIN, config.MMN_TMAX, [fz_ch]),

            "mmn_A_amp_ROI":      _amp(mmn_A_e,  config.MMN_TMIN, config.MMN_TMAX, config.ROI_MMN),
            "mmn_B_amp_ROI":      _amp(mmn_B_e,  config.MMN_TMIN, config.MMN_TMAX, config.ROI_MMN),

            "mmn_A_fal_ms":       _fal(mmn_A_e,  config.MMN_LATENCY_TMIN, config.MMN_LATENCY_TMAX, fz_ch, "negative"),
            "mmn_B_fal_ms":       _fal(mmn_B_e,  config.MMN_LATENCY_TMIN, config.MMN_LATENCY_TMAX, fz_ch, "negative"),


            "p3a_A_amp_Cz":       _amp(mmn_A_e,  config.P3A_TMIN, config.P3A_TMAX, [cz_ch]),
            "p3a_B_amp_Cz":       _amp(mmn_B_e,  config.P3A_TMIN, config.P3A_TMAX, [cz_ch]),
            "p3a_A_amp_ROI":      _amp(mmn_A_e,  config.P3A_TMIN, config.P3A_TMAX, config.ROI_P3A),
            "p3a_B_amp_ROI":      _amp(mmn_B_e,  config.P3A_TMIN, config.P3A_TMAX, config.ROI_P3A),

            "p3a_A_fal_ms":       _fal(mmn_A_e,  config.P3A_TMIN, config.P3A_TMAX, cz_ch, "positive"),
            "p3a_B_fal_ms":       _fal(mmn_B_e,  config.P3A_TMIN, config.P3A_TMAX, cz_ch, "positive"),

            "ssa_AC_amp_Fz":      _amp(ssa_AC_e, config.MMN_TMIN, config.MMN_TMAX, [fz_ch]),
            "ssa_BC_amp_Fz":      _amp(ssa_BC_e, config.MMN_TMIN, config.MMN_TMAX, [fz_ch]),

            "n_trials_std_A":     erp[p]["_n_std_A"],
            "n_trials_dev_A":     erp[p]["_n_dev_A"],
            "n_trials_std_B":     erp[p]["_n_std_B"],
            "n_trials_dev_B":     erp[p]["_n_dev_B"],
            "n_trials_ctrl_C":    erp[p]["_n_ctrl_C"],
        }
        records.append(row)

    df = pd.DataFrame(records)
    csv_path = Path(config.STATS_PATH) / "erp_measures.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")
    log.info("Saved ERP measures: %s", csv_path)

    ga_all = {
        "std_A":  ga_std_A,
        "dev_A":  ga_dev_A,
        "std_B":  ga_std_B,
        "dev_B":  ga_dev_B,
        "ctrl_C": ga_ctrl_C,
    }
    sem_all = {
        "std_A":  [erp[p]["std_A"]  for p in participants if erp[p]["std_A"]  is not None],
        "dev_A":  [erp[p]["dev_A"]  for p in participants if erp[p]["dev_A"]  is not None],
        "std_B":  [erp[p]["std_B"]  for p in participants if erp[p]["std_B"]  is not None],
        "dev_B":  [erp[p]["dev_B"]  for p in participants if erp[p]["dev_B"]  is not None],
        "ctrl_C": [erp[p]["ctrl_C"] for p in participants if erp[p]["ctrl_C"] is not None],
    }


    plot_grand_avg_all_conditions(ga_all, sem_all)


    plot_mmn_difference_waves(
        ga_mmn_A, ga_mmn_B,
        [mmn_per_p[p] for p in participants if mmn_per_p.get(p) is not None],
        [mmn_B_per_p[p] for p in participants if mmn_B_per_p.get(p) is not None],
    )


    plot_topomap_mmn(ga_mmn_A, "A")
    plot_topomap_mmn(ga_mmn_B, "B")
    plot_topomap_mmn_combined(ga_mmn_A, ga_mmn_B)


    plot_topomap_p3a(ga_mmn_A, ga_mmn_B)
    plot_topomap_p3a_combined(ga_mmn_A, ga_mmn_B)


    plot_ssa_control(
        ga_ssa_AC, ga_ssa_BC,
        [ssa_AC_per_p[p] for p in participants if ssa_AC_per_p.get(p) is not None],
        [ssa_BC_per_p[p] for p in participants if ssa_BC_per_p.get(p) is not None],
    )


    plot_individual_erps(
        {p: {k: erp[p][k] for k in ["std_A", "dev_A", "std_B", "dev_B", "ctrl_C"]} for p in participants}
    )

    log.info("=" * 60)
    log.info("02_erp_analysis.py – FINISHED SUCCESSFULLY")
    log.info("=" * 60)


if __name__ == "__main__":
    main()