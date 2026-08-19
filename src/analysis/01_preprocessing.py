import sys
import json
import shutil
import traceback
import importlib.metadata
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import mne

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
mne.set_log_level('WARNING')
mne.viz.set_browser_backend('matplotlib')

MANUAL_REVIEW_FLAGS: list[dict] = []


def _to_python(obj):
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


def print_library_versions() -> None:
    libs = ['mne', 'numpy', 'scipy', 'sklearn']
    print("\n=== Library versions ===")
    for lib in libs:
        try:
            ver = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            ver = 'not found'
        print(f"  {lib:10s}: {ver}")
    print()


def _read_internal_filenames(vhdr_path: Path) -> dict:
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

def find_cleaned_vmrk(participant_id: str, block_letter: str) -> Path | None:
    pattern = f"{participant_id}{block_letter}.vmrk"
    matches = list(cfg.VMRK_CLEANED_PATH.glob(pattern))
    if not matches:
        pattern_lower = f"{participant_id.lower()}_{block_letter}.vmrk"
        matches = list(cfg.VMRK_CLEANED_PATH.glob(pattern_lower))
    return matches[0] if matches else None

def copy_files_to_tmp(files_dict: dict,
                      participant_id: str,
                      block_letter: str,
                      tmp_dir: Path) -> Path:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    vhdr_src = files_dict['vhdr']

    internal = _read_internal_filenames(vhdr_src)
    eeg_internal = internal['eeg'] or (vhdr_src.stem + '.eeg')
    vmrk_internal = internal['vmrk'] or (vhdr_src.stem + '.vmrk')

    print(f"    .vhdr DataFile  : '{eeg_internal}'")
    print(f"    .vhdr MarkerFile: '{vmrk_internal}'")

    vhdr_dst = tmp_dir / vhdr_src.name
    shutil.copy2(vhdr_src, vhdr_dst)

    eeg_src = files_dict.get('eeg')
    if eeg_src and eeg_src.exists():
        shutil.copy2(eeg_src, tmp_dir / eeg_internal)
        print(f"    eeg_data/{eeg_src.name} -> tmp/{eeg_internal}")
    else:
        print(f"    WARNING: Missing .eeg file ({vhdr_src.stem})")

    cleaned_vmrk = files_dict.get('vmrk_cleaned')
    if cleaned_vmrk and cleaned_vmrk.exists():
        shutil.copy2(cleaned_vmrk, tmp_dir / vmrk_internal)
        print(f"    vmrk_cleaned/{cleaned_vmrk.name} -> tmp/{vmrk_internal}")
    else:
        print(f"    ERROR: Missing cleaned .vmrk for {participant_id} block {block_letter.upper()}")
        raise FileNotFoundError(f"Missing cleaned .vmrk")

    return vhdr_dst

def cleanup_tmp(tmp_dir: Path, vhdr_name: str) -> None:
    stem = Path(vhdr_name).stem
    for ext in ['.vhdr', '.eeg']:
        f = tmp_dir / f"{stem}{ext}"
        if f.exists():
            try:
                f.unlink()
            except OSError:
                pass
    for vmrk in tmp_dir.glob('*.vmrk'):
        try:
            vmrk.unlink()
        except OSError:
            pass


def load_raw(vhdr_path: Path) -> tuple[mne.io.Raw, dict]:
    print(f"\n  [1] Loading: {vhdr_path.name}")
    raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True)
    print(f"      sfreq       : {raw.info['sfreq']} Hz")
    print(f"      n_channels  : {raw.info['nchan']}")
    print(f"      duration    : {raw.times[-1]:.1f} s")

    events, event_id = mne.events_from_annotations(raw)
    print(f"      event_id    : {event_id}")
    print(f"      n_events    : {len(events)}")
    return raw, event_id


def resample_raw(raw: mne.io.Raw) -> mne.io.Raw:
    sfreq_orig = raw.info['sfreq']
    if sfreq_orig != cfg.SFREQ_TARGET:
        print(f"  [2] Resample: {sfreq_orig} Hz -> {cfg.SFREQ_TARGET} Hz")
        raw.resample(cfg.SFREQ_TARGET, npad='auto')
    else:
        print(f"  [2] Resample: skipped (already {cfg.SFREQ_TARGET} Hz)")
    return raw


def setup_channels(raw: mne.io.Raw) -> mne.io.Raw:
    print("  [3] Montage and channel verification...")
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='warn')

    present = [ch.upper() for ch in raw.ch_names]
    for ch in cfg.ALL_ROI_CHANNELS:
        if ch.upper() not in present:
            print(f"      WARNING: ROI channel '{ch}' not found in data!")

    print(f"      ROI channels OK: "
          f"{[ch for ch in cfg.ALL_ROI_CHANNELS if ch.upper() in present]}")

    missing = [ch for ch in cfg.ALL_ROI_CHANNELS if ch.upper() not in present]
    if missing:
        print(f"      ROI channels MISSING: {missing}")

    return raw


def detect_and_fix_bridges(raw: mne.io.Raw) -> list[tuple[str, str]]:
    """Detects salt bridges via the electrical distance method
    (Alschuler et al., 2014; Tenke & Kayser, 2001) and repairs bridged
    pairs by interpolating both from the remaining electrodes."""
    print("  [3.4] Detecting salt bridges (electrical distance)...")

    try:
        bridged_idx, ed_matrix = mne.preprocessing.compute_bridged_electrodes(raw)
    except Exception as e:
        print(f"      WARNING: bridge detection failed: {e}")
        return []

    if not bridged_idx:
        print("      No salt bridges detected")
        return []

    pairs = [(raw.ch_names[i], raw.ch_names[j]) for i, j in bridged_idx]
    print(f"      Detected bridges: {pairs}")

    try:
        mne.preprocessing.interpolate_bridged_electrodes(raw, bridged_idx=bridged_idx)
        print("      Bridges repaired (virtual interpolation of both channels)")
    except Exception as e:
        print(f"      WARNING: automatic bridge repair failed ({e}), "
              f"marking as bad and interpolating standard way...")
        bads = sorted(set(ch for pair in pairs for ch in pair))
        raw.info['bads'] = sorted(set(raw.info['bads']) | set(bads))
        raw.interpolate_bads(reset_bads=True)

    return pairs


def detect_bad_channels(raw: mne.io.Raw, interpolate: bool = True) -> tuple[list[str], dict]:
    """Flags bad channels by extreme variance and by correlation with the
    rest of the montage (robust z relative to the montage, not a fixed
    threshold, since distant electrodes on this sparse 32-ch cap naturally
    correlate weakly). Returns (bad_channels, review_info); review_info
    flags cases above MAX_BAD_CH_FRACTION, where the list is truncated and
    manual review is required."""
    print("  [3.5] Bad channel detection...")

    MAX_BAD_CH_FRACTION = 0.25

    eeg_data = raw.get_data(picks='eeg')
    ch_names_eeg = raw.copy().pick('eeg').ch_names
    n_channels, n_times = eeg_data.shape

    variances = np.var(eeg_data, axis=1)
    median_var = np.median(variances)
    mad_var = np.median(np.abs(variances - median_var))

    bad_channels = []
    bad_reasons = {}

    threshold = median_var + cfg.BAD_CH_THRESHOLD_STD * mad_var
    high_var_channels = np.where(variances > threshold)[0]
    for idx in high_var_channels:
        ch_name = ch_names_eeg[idx]
        print(f"      Bad channel (high variance): {ch_name}")
        bad_channels.append(ch_name)
        bad_reasons[ch_name] = 'high_variance'

    corr_matrix = np.corrcoef(eeg_data)
    per_channel_median_corr = np.zeros(n_channels)
    for i in range(n_channels):
        correlations = np.abs(corr_matrix[i])
        mask = np.ones(n_channels, dtype=bool)
        mask[i] = False
        per_channel_median_corr[i] = np.median(correlations[mask])

    corr_center = np.median(per_channel_median_corr)
    corr_mad = np.median(np.abs(per_channel_median_corr - corr_center)) + 1e-12
    corr_robust_z = (per_channel_median_corr - corr_center) / (1.4826 * corr_mad)

    CORR_Z_THRESHOLD = -4.0
    for i in range(n_channels):
        if i in high_var_channels:
            continue
        if corr_robust_z[i] < CORR_Z_THRESHOLD:
            ch_name = ch_names_eeg[i]
            print(f"      Bad channel (outlying correlation): {ch_name} "
                  f"(median corr={per_channel_median_corr[i]:.3f}, z={corr_robust_z[i]:.1f})")
            if ch_name not in bad_channels:
                bad_channels.append(ch_name)
                bad_reasons[ch_name] = 'low_relative_correlation'

    max_allowed = max(1, int(round(MAX_BAD_CH_FRACTION * n_channels)))
    n_detected_before_truncation = len(bad_channels)
    manual_review_required = n_detected_before_truncation > max_allowed
    if manual_review_required:
        print(f"      *** WARNING: algorithm marked {len(bad_channels)}/{n_channels} channels as bad "
              f"(> {MAX_BAD_CH_FRACTION*100:.0f}% of montage). Truncating to {max_allowed} worst "
              f"channels; the rest remain uninterpolated - manual review required. ***")
        ranked = sorted(
            bad_channels,
            key=lambda ch: corr_robust_z[ch_names_eeg.index(ch)] if ch in ch_names_eeg else 0
        )
        bad_channels = ranked[:max_allowed]

    if bad_channels:
        print(f"      Total bad channels detected: {len(bad_channels)}")
        raw.info['bads'] = bad_channels
        if interpolate:
            print(f"      Interpolating bad channels...")
            raw.interpolate_bads(reset_bads=True)
        else:
            print(f"      Interpolation DEFERRED (ica_before_interpolate strategy) - "
                  f"will interpolate after ICA is applied.")
    else:
        print("      No bad channels detected")

    review_info = {
        'manual_review_required': manual_review_required,
        'n_bad_detected_before_truncation': n_detected_before_truncation,
        'n_bad_interpolated': len(bad_channels),
    }
    return bad_channels, review_info


def filter_raw(raw: mne.io.Raw) -> mne.io.Raw:
    print(f"  [4] Notch filter {cfg.NOTCH_FREQ} Hz + "
          f"bandpass {cfg.L_FREQ}–{cfg.H_FREQ} Hz...")
    raw.notch_filter(
        freqs=[cfg.NOTCH_FREQ, cfg.NOTCH_FREQ * 2],
        picks='eeg',
        method='fir',
        fir_window='hamming',
    )
    raw.filter(
        l_freq=cfg.L_FREQ,
        h_freq=cfg.H_FREQ,
        picks='eeg',
        fir_window='hamming',
        filter_length='auto',
        phase='zero',
    )
    return raw


def _sphere_covering_all_channels(info: mne.Info, margin: float = 1.03) -> tuple:
    """Explicit (x, y, z, radius) sphere for plot_topomap/plot_components,
    sized to cover the most eccentric channel, unlike MNE's default
    sphere='auto' fit."""
    pos = np.array([ch['loc'][:3] for ch in info['chs'] if ch['kind'] == 2])  # FIFFV_EEG_CH
    center_xy = pos[:, :2].mean(axis=0)
    radii = np.sqrt(((pos[:, :2] - center_xy) ** 2).sum(axis=1))
    r = float(radii.max()) * margin
    return (float(center_xy[0]), float(center_xy[1]), 0.0, r)


def run_ica(raw: mne.io.Raw,
            participant_id: str,
            block: str,
            random_state: int = cfg.RANDOM_STATE,
            n_components_override: int | None = None,
            exclude_from_fit: list[str] | None = None) -> mne.preprocessing.ICA:
    """n_components_override overrides cfg.N_ICA_COMPONENTS (used by the
    reduce_ica_components strategy). exclude_from_fit lists channels left
    out of the fit entirely (used by ica_before_interpolate)."""
    n_components = n_components_override if n_components_override is not None else cfg.N_ICA_COMPONENTS

    raw_for_ica = raw.copy().filter(1.0, None, picks='eeg', fir_window='hamming')

    fit_picks = 'eeg'
    if exclude_from_fit:
        fit_picks = [ch for ch in raw_for_ica.copy().pick('eeg').ch_names
                     if ch not in exclude_from_fit]
        # n_components cannot exceed len(fit_picks)-1
        max_components = len(fit_picks) - 1
        if n_components > max_components:
            print(f"      Capping n_components {n_components} -> {max_components} "
                  f"(len(fit_picks)={len(fit_picks)} after excluding "
                  f"{len(exclude_from_fit)} not-yet-interpolated bad channel(s))")
            n_components = max_components
        print(f"      Excluding {len(exclude_from_fit)} not-yet-interpolated "
              f"bad channel(s) from ICA fit: {exclude_from_fit}")

    print(f"  [5] ICA (n_components={n_components}, "
          f"random_state={random_state})...")

    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method='fastica',
        random_state=random_state,
        max_iter=1000,
    )
    ica.fit(raw_for_ica, picks=fit_picks)
    print("      ICA fit OK")

    eog_proxy = next((ch for ch in ['Fp1', 'Fp2'] if ch in raw.ch_names), None)
    if eog_proxy:
        print(f"      EOG detection via proxy: {eog_proxy}")
        eog_indices, _ = ica.find_bads_eog(raw, ch_name=eog_proxy, threshold=3.0)
        ica.exclude = eog_indices
        print(f"      EOG components to remove: {[int(i) for i in eog_indices]}")
    else:
        print("      WARNING: No Fp1/Fp2 – skipping auto-detection of EOG.")

    fig_dir = cfg.FIGURES_INDIVIDUAL_PATH
    fig_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{participant_id}_block{block}"

    try:
        sphere = _sphere_covering_all_channels(raw_for_ica.info)
        figs = ica.plot_components(show=False, sphere=sphere)
        if not isinstance(figs, list):
            figs = [figs]
        for i, fig in enumerate(figs):
            out = fig_dir / f"{stem}_ica_components_{i}.png"
            fig.savefig(out, dpi=150, bbox_inches='tight')
            plt.close(fig)
        print(f"      ICA topomap -> {fig_dir.name}/")
    except Exception as e:
        print(f"      WARNING: ICA topomap not saved: {e}")

    try:
        fig_src = ica.plot_sources(raw_for_ica, picks=list(range(min(15, ica.n_components_))),
                                   show=False, start=0, stop=30)
        if hasattr(fig_src, 'savefig'):
            out_src = fig_dir / f"{stem}_ica_sources.png"
            fig_src.savefig(out_src, dpi=150, bbox_inches='tight')
            plt.close(fig_src)
            print(f"      ICA sources    -> {out_src.name}")
        else:
            print("      WARNING: plot_sources still returned a non-figure "
                  "object – skipping save.")
            plt.close('all')
    except Exception as e:
        print(f"      WARNING: ICA sources not saved: {e}")

    return ica


def rereference(raw: mne.io.Raw) -> mne.io.Raw:
    """Re-reference to the common average of all EEG channels."""
    print("  [6] Re-referencing: average of all EEG channels")
    raw.set_eeg_reference(ref_channels="average", projection=False)
    return raw


def _remap_trigger_codes(event_id: dict) -> dict:
    remap = {}
    for semantic_key, expected_int in cfg.TRIGGER_CODES_INT.items():
        found = next(
            (ann_str for ann_str, val in event_id.items() if val == expected_int),
            None
        )
        if found:
            remap[semantic_key] = found
        else:
            print(f"      WARNING: Trigger int={expected_int} not found "
                  f"({semantic_key}) in event_id: {dict(event_id)}")
    return remap

def _find_pre_deviant_standards(events_std: np.ndarray,
                                 events_dev: np.ndarray,
                                 sfreq: float,
                                 soa_ms: float = 900.0) -> np.ndarray:
    soa_samples = int(soa_ms * sfreq / 1000)
    tolerance = int(0.05 * sfreq)
    dev_onsets = events_dev[:, 0]
    pre_dev_mask = np.array([
        np.any(np.abs(dev_onsets - (onset + soa_samples)) <= tolerance)
        for onset in events_std[:, 0]
    ], dtype=bool)
    n = pre_dev_mask.sum()
    if n:
        print(f"      Excluded {n} standards immediately before a deviant")
    return pre_dev_mask


def epoch_block(raw: mne.io.Raw,
                events: np.ndarray,
                event_id: dict,
                block: str,
                trigger_remap: dict) -> dict:
    LEAD_IN = cfg.LEAD_IN
    result = {'std': None, 'dev': None}
    sfreq = raw.info['sfreq']

    def get_events(semantic_key):
        if semantic_key not in trigger_remap:
            return None
        ann_str = trigger_remap[semantic_key]
        code = event_id[ann_str]
        ev = events[events[:, 2] == code].copy()
        return ev if len(ev) > 0 else None

    def apply_offset_to_events(ev, code):
        if ev is None or len(ev) == 0:
            return ev
        offset_sec = cfg.TRIGGER_OFFSETS.get(code, 0.0)
        if offset_sec == 0.0 and code not in cfg.TRIGGER_OFFSETS:
            print(f"      WARNING: No offset defined for trigger code {code} – using 0")
        offset_samples = int(offset_sec * sfreq)
        ev[:, 0] += offset_samples
        return ev

    def make_epochs(ev, ann_str):
        eid = {ann_str: event_id[ann_str]}
        return mne.Epochs(
            raw, events=ev, event_id=eid,
            tmin=cfg.EPOCH_TMIN, tmax=cfg.EPOCH_TMAX,
            baseline=cfg.BASELINE,
            preload=True, picks='eeg', verbose=False,
        )

    if block == 'A':
        std_key, dev_key = 'A_standard', 'A_deviant'
    elif block == 'B':
        std_key, dev_key = 'B_standard', 'B_deviant'
    else:
        std_key, dev_key = 'C_control', None

    # --- Standard ---
    ev_std = get_events(std_key)
    if ev_std is not None and len(ev_std) > LEAD_IN:
        ev_std = ev_std[LEAD_IN:]
        code_std = event_id[trigger_remap[std_key]]
        ev_std = apply_offset_to_events(ev_std, code_std)

        if cfg.EXCLUDE_PRE_DEVIANT_STANDARDS and dev_key:
            ev_dev_all = get_events(dev_key)
            if ev_dev_all is not None:
                mask = _find_pre_deviant_standards(ev_std, ev_dev_all, sfreq)
                ev_std = ev_std[~mask]

        result['std'] = make_epochs(ev_std, trigger_remap[std_key])
        print(f"      Standard [{std_key}]: {len(ev_std)} epochs before AR (offset applied)")
    elif ev_std is not None:
        print(f"      WARNING: Not enough standards after lead-in ({len(ev_std)})")

    # LEAD_IN standards before the first deviant in the block are excluded
    # above; deviants themselves are not trimmed by lead-in.
    if dev_key:
        ev_dev = get_events(dev_key)
        if ev_dev is not None and len(ev_dev) > 0:
            code_dev = event_id[trigger_remap[dev_key]]
            ev_dev = apply_offset_to_events(ev_dev, code_dev)
            result['dev'] = make_epochs(ev_dev, trigger_remap[dev_key])
            print(f"      Deviant  [{dev_key}]: {len(ev_dev)} epochs before AR (offset applied, "
                  f"NO lead-in trim)")

    return result


def reject_artifacts(epochs_dict: dict,
                     participant_id: str,
                     block: str) -> dict:
    stats = {}
    for cond, epochs in epochs_dict.items():
        if epochs is None:
            continue

        n_before = len(epochs)
        epochs.drop_bad(reject={'eeg': cfg.REJECT_THRESHOLD})
        n_after = len(epochs)
        n_rejected = n_before - n_after
        pct = 100.0 * n_rejected / n_before if n_before > 0 else 0.0

        print(f"      AR [{cond}]: {n_before} -> {n_after} "
              f"(rejected {n_rejected}, {pct:.1f}%)")

        stats[cond] = {
            'n_before': n_before, 'n_after': n_after,
            'n_rejected': n_rejected, 'pct_rejected': round(pct, 1),
        }

        if cond == 'dev' and n_after < cfg.MIN_DEVIANT_TRIALS:
            print(f"      WARNING: {participant_id} block {block}: "
                  f"only {n_after} deviant trials after AR "
                  f"(minimum: {cfg.MIN_DEVIANT_TRIALS})!")

    return stats


def save_epochs(epochs_dict: dict,
                ar_stats: dict,
                ica,
                participant_id: str,
                block: str,
                sfreq_original: float,
                bad_channels: list[str] | None = None,
                bridged_pairs: list[tuple[str, str]] | None = None,
                review_info: dict | None = None) -> None:
    out_dir = cfg.PREPROCESSED_PATH
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{participant_id}_block{block}"

    report = {
        'participant': participant_id,
        'block': block,
        'sfreq_original': float(sfreq_original),
        'ica_components_removed': [int(i) for i in (ica.exclude if ica else [])],
        'bad_channels': list(bad_channels) if bad_channels else [],
        'bridged_electrode_pairs': [list(p) for p in bridged_pairs] if bridged_pairs else [],
        'manual_review_required': bool(review_info.get('manual_review_required', False)) if review_info else False,
        'n_bad_detected_before_truncation': review_info.get('n_bad_detected_before_truncation') if review_info else None,
    }

    for cond, epochs in epochs_dict.items():
        if epochs is None:
            continue
        fif_path = out_dir / f"{stem}_{cond}_epo.fif"
        epochs.save(str(fif_path), overwrite=True)
        print(f"      Saved: {fif_path.name}")

        if cond in ar_stats:
            s = ar_stats[cond]
            report.update({
                f'n_{cond}_before': s['n_before'],
                f'n_{cond}_after': s['n_after'],
                f'n_{cond}_rejected': s['n_rejected'],
                f'pct_{cond}_rejected': s['pct_rejected'],
            })

    json_path = out_dir / f"{stem}_report.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(_to_python(report), f, indent=2, ensure_ascii=False)
    print(f"      JSON report: {json_path.name}")


def preprocess_participant(participant_id: str) -> dict:
    print(f"\n{'='*70}")
    print(f"  PARTICIPANT: {participant_id}")
    print(f"{'='*70}")

    block_results = {}
    files = cfg.get_participant_files(participant_id)

    # Blocks with no matching .vhdr file are marked MISSING rather than ERROR
    for _letter, _blk in (('a', 'A'), ('b', 'B'), ('c', 'C')):
        if _blk not in files:
            block_results[_blk] = 'MISSING'

    if not files:
        print(f"  ERROR: No files for {participant_id}.")
        return {'A': 'ERROR', 'B': 'ERROR', 'C': 'ERROR'}

    for block_type, files_dict in files.items():
        block_letter = block_type.lower()
        print(f"\n  --- Block {block_type} ---")

        if cfg.is_excluded(participant_id, block_type):
            print(f"  EXCLUDED: {participant_id} block {block_type} "
                  f"(see results/tables/excluded_recordings.csv)")
            block_results[block_type] = 'EXCLUDED'
            continue

        try:
            tmp_vhdr = copy_files_to_tmp(
                files_dict, participant_id, block_letter, cfg.TMP_PATH
            )
            raw, event_id = load_raw(tmp_vhdr)
            sfreq_original = raw.info['sfreq']

            raw = setup_channels(raw)
            raw = filter_raw(raw)
            raw = resample_raw(raw)

            # Bridges are detected before standard bad-channel detection,
            # since a bridge would distort the variance/correlation checks
            bridged_pairs = detect_and_fix_bridges(raw)

            defer_interpolation = (cfg.ICA_STRATEGY == 'ica_before_interpolate')
            bad_channels, review_info = detect_bad_channels(raw, interpolate=not defer_interpolation)

            ica_n_components_override = None
            if cfg.ICA_STRATEGY == 'reduce_ica_components' and bad_channels:
                ica_n_components_override = max(
                    cfg.MIN_ICA_COMPONENTS,
                    cfg.N_ICA_COMPONENTS - len(bad_channels)
                )
                print(f"      ICA_STRATEGY=reduce_ica_components: "
                      f"{len(bad_channels)} interpolated channel(s) -> "
                      f"n_components {cfg.N_ICA_COMPONENTS} -> {ica_n_components_override}")

            ica_exclude_from_fit = bad_channels if defer_interpolation else None

            try:
                ica = run_ica(raw, participant_id, block_type,
                               n_components_override=ica_n_components_override,
                               exclude_from_fit=ica_exclude_from_fit)
            except Exception as ica_err:
                print(f"      WARNING: ICA (rs={cfg.RANDOM_STATE}) failed to converge: {ica_err}")
                print("      Retrying with random_state=123, max_iter=2000...")
                try:
                    raw_fb = raw.copy().filter(1.0, None, picks='eeg')
                    fallback_picks = 'eeg'
                    fallback_n_components = ica_n_components_override or cfg.N_ICA_COMPONENTS
                    if ica_exclude_from_fit:
                        fallback_picks = [ch for ch in raw_fb.copy().pick('eeg').ch_names
                                           if ch not in ica_exclude_from_fit]
                        fallback_n_components = min(fallback_n_components, len(fallback_picks) - 1)
                    ica = mne.preprocessing.ICA(
                        n_components=fallback_n_components,
                        method='fastica', random_state=123, max_iter=2000,
                    )
                    ica.fit(raw_fb, picks=fallback_picks)
                    print("      ICA (fallback) OK")
                except Exception as ica_err2:
                    print(f"      ERROR: ICA fallback also failed to converge: {ica_err2}")
                    ica = None

            if ica is not None and ica.exclude:
                ica.apply(raw)
                print(f"      ICA applied, removed: {[int(i) for i in ica.exclude]}")
            elif ica is not None:
                print("      ICA: no components to remove")

            if defer_interpolation and bad_channels:
                print(f"      Interpolating bad channels (post-ICA): {bad_channels}")
                raw.info['bads'] = bad_channels
                raw.interpolate_bads(reset_bads=True)

            raw = rereference(raw)

            print("  [7] Epoching...")
            events, event_id_up = mne.events_from_annotations(raw)
            trigger_remap = _remap_trigger_codes(event_id_up)
            epochs_dict = epoch_block(raw, events, event_id_up, block_type, trigger_remap)

            print("  [8] Artifact rejection...")
            ar_stats = reject_artifacts(epochs_dict, participant_id, block_type)

            print("  [9] Saving...")
            save_epochs(epochs_dict, ar_stats, ica,
                        participant_id, block_type, sfreq_original,
                        bad_channels=bad_channels, bridged_pairs=bridged_pairs,
                        review_info=review_info)

            if review_info.get('manual_review_required'):
                MANUAL_REVIEW_FLAGS.append({
                    'participant': participant_id,
                    'block': block_type,
                    'n_bad_detected_before_truncation': review_info['n_bad_detected_before_truncation'],
                    'n_bad_interpolated': review_info['n_bad_interpolated'],
                    'n_channels_total': 32,
                })

            block_results[block_type] = 'OK'
            print(f"  Block {block_type}: OK")

        except FileNotFoundError as e:
            print(f"  Block {block_type}: Missing file – {e}")
            block_results[block_type] = 'ERROR'
        except Exception as e:
            print(f"  Block {block_type}: UNEXPECTED ERROR")
            traceback.print_exc()
            block_results[block_type] = 'ERROR'
        finally:
            if files_dict.get('vhdr'):
                cleanup_tmp(cfg.TMP_PATH, files_dict['vhdr'].name)

    return block_results


def main():
    print_library_versions()
    cfg.ensure_output_dirs()

    if not cfg.VMRK_CLEANED_PATH.exists():
        print(f"  WARNING: Folder vmrk_cleaned does not exist: {cfg.VMRK_CLEANED_PATH}")
        print(f"            Make sure cleaned .vmrk files are there!")
    else:
        vmrk_files = list(cfg.VMRK_CLEANED_PATH.glob('*.vmrk'))
        print(f"  vmrk_cleaned/: found {len(vmrk_files)} .vmrk files")

    all_results = {}
    for pid in cfg.PARTICIPANTS:
        try:
            all_results[pid] = preprocess_participant(pid)
        except Exception as e:
            print(f"\nCRITICAL ERROR for {pid}: {e}")
            traceback.print_exc()
            all_results[pid] = {'A': 'ERROR', 'B': 'ERROR', 'C': 'ERROR'}

    print(f"\n{'='*70}")
    print("  PREPROCESSING SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Participant':<10} {'Block A':<12} {'Block B':<12} {'Block C':<12}")
    print(f"  {'-'*46}")

    icons = {'OK': 'OK', 'WARNING': 'WARN', 'ERROR': 'ERR', 'EXCLUDED': 'EXCL', 'MISSING': 'N/A'}
    for pid, blocks in all_results.items():
        row = f"  {pid:<10}"
        for blk in ['A', 'B', 'C']:
            st = blocks.get(blk, 'ERROR')
            row += f" {icons.get(st, '?')} {st:<10}"
        print(row)

    print(f"\n  Legend: OK = OK, WARN = Warning, ERR = Error, "
          f"EXCL = Excluded (see excluded_recordings.csv), N/A = No recording exists")
    print(f"{'='*70}\n")

    summary_path = cfg.REPORTS_PATH / 'preprocessing_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"  Summary -> {summary_path}")

    if MANUAL_REVIEW_FLAGS:
        import csv
        review_path = cfg.STATS_PATH / 'manual_review_flags.csv'
        cfg.STATS_PATH.mkdir(parents=True, exist_ok=True)
        with open(review_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(MANUAL_REVIEW_FLAGS[0].keys()))
            writer.writeheader()
            writer.writerows(MANUAL_REVIEW_FLAGS)
        print(f"  Manual review flags ({len(MANUAL_REVIEW_FLAGS)} blocks) -> {review_path}")
    else:
        print("  No blocks required manual review.")

if __name__ == '__main__':
    main()
    try:
        from qtpy.QtWidgets import QApplication
        _app = QApplication.instance()
        if _app is not None:
            _app.quit()
    except ImportError:
        pass
    sys.exit(0)