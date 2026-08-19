"""
manual_eeg_browser.py  –  Interactive EEG browser for manual artifact inspection

Purpose:
    Load raw EEG data (or preprocessed epochs) for a given participant and block,
    and provide interactive MNE plots to manually inspect for artifacts.

Usage examples:
    # Browse raw data for participant P01, block A (with filtering and montage)
    python src/analysis/manual_eeg_browser.py --participant P01 --block A

    # Load preprocessed epochs directly (skip raw preprocessing)
    python src/analysis/manual_eeg_browser.py --participant P01 --block A --epochs-only

    # Run ICA on raw data and browse components
    python src/analysis/manual_eeg_browser.py --participant P01 --block A --ica

    # Load raw data without any preprocessing (only montage)
    python src/analysis/manual_eeg_browser.py --participant P01 --block A --raw-only

    # Use a different random state for ICA (fallback if default fails)
    python src/python analysis/manual_eeg_browser.py --participant P01 --block A --ica --ica-rs 123

Interactive controls (MNE built-in):
    - Raw browser: scroll, mark bad segments with 'b', annotate with 'a'
    - Epochs browser: click epochs to mark as bad, 'd' to drop
    - ICA: inspect components, click to select/deselect for removal
"""

import sys
import argparse
from pathlib import Path

import mne
import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory to path so config can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

mne.set_log_level('WARNING')


def _sphere_covering_all_channels(info: mne.Info, margin: float = 1.03) -> tuple:
    """Explicit (x, y, z, radius) sphere for plot_topomap's `sphere`
    argument, sized to cover the most eccentric channel (default
    sphere='auto' can leave FT9/FT10/TP9/TP10 outside the drawn head
    circle on this 32-ch montage). Purely cosmetic - doesn't affect
    data or analysis. Same fix as in 02_erp_analysis.py."""
    pos = np.array([ch['loc'][:3] for ch in info['chs'] if ch['kind'] == 2])  # FIFFV_EEG_CH
    center_xy = pos[:, :2].mean(axis=0)
    radii = np.sqrt(((pos[:, :2] - center_xy) ** 2).sum(axis=1))
    r = float(radii.max()) * margin
    return (float(center_xy[0]), float(center_xy[1]), 0.0, r)


# --- Helper functions (reused from preprocessing pipeline) ---

def _read_internal_filenames(vhdr_path: Path) -> dict:
    """Read .vhdr file and extract internal DataFile= and MarkerFile= names."""
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


def _find_cleaned_vmrk(participant_id: str, block_letter: str) -> Path | None:
    """Search for cleaned .vmrk in VMRK_CLEANED_PATH."""
    pattern = f"{participant_id}_*_{block_letter}.vmrk"
    matches = list(cfg.VMRK_CLEANED_PATH.glob(pattern))
    if not matches:
        pattern_lower = f"{participant_id.lower()}_*_{block_letter}.vmrk"
        matches = list(cfg.VMRK_CLEANED_PATH.glob(pattern_lower))
    return matches[0] if matches else None


def load_raw_participant_block(participant_id: str, block: str) -> mne.io.Raw:
    """
    Load raw EEG data for a given participant and block.
    Uses the same logic as the preprocessing pipeline.
    """
    block_letter = block.lower()
    files = cfg.get_participant_files(participant_id)

    if block not in files:
        raise ValueError(f"Block {block} not found for participant {participant_id}")

    file_dict = files[block]
    vhdr_path = file_dict['vhdr']

    # Find cleaned .vmrk and copy to tmp if needed
    cleaned_vmrk = _find_cleaned_vmrk(participant_id, block_letter)
    if cleaned_vmrk and cleaned_vmrk.exists():
        # Use the cleaned .vmrk by temporarily setting it as the marker file
        # Actually, we need to copy it to the same location as the .vhdr for MNE to find it
        import shutil
        tmp_dir = cfg.TMP_PATH
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # Read internal filenames
        internal = _read_internal_filenames(vhdr_path)
        vmrk_internal = internal['vmrk'] or (vhdr_path.stem + '.vmrk')

        # Copy .vhdr to tmp
        vhdr_dst = tmp_dir / vhdr_path.name
        shutil.copy2(vhdr_path, vhdr_dst)

        # Copy .eeg to tmp
        eeg_src = file_dict.get('eeg')
        if eeg_src and eeg_src.exists():
            eeg_internal = internal['eeg'] or (vhdr_path.stem + '.eeg')
            shutil.copy2(eeg_src, tmp_dir / eeg_internal)

        # Copy cleaned .vmrk to tmp with the expected name
        shutil.copy2(cleaned_vmrk, tmp_dir / vmrk_internal)

        # Load from tmp
        raw = mne.io.read_raw_brainvision(str(vhdr_dst), preload=True)
        print(f"  Loaded: {vhdr_path.name} (with cleaned .vmrk)")
        return raw
    else:
        # Fallback: load directly from the original files
        raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True)
        print(f"  Loaded: {vhdr_path.name} (using original .vmrk)")
        return raw


def load_epochs_participant_block(participant_id: str, block: str, condition: str = 'dev'):
    """
    Load preprocessed epochs for a given participant, block, and condition.
    """
    fname = cfg.PREPROCESSED_PATH / f"{participant_id}_block{block}_{condition}_epo.fif"
    if not fname.exists():
        print(f"  ✗ File not found: {fname}")
        return None
    epochs = mne.read_epochs(fname, preload=True, verbose=False)
    print(f"  Loaded: {fname.name}  ({len(epochs)} epochs)")
    return epochs


def load_ica_from_raw(raw: mne.io.Raw, random_state: int = cfg.RANDOM_STATE) -> mne.preprocessing.ICA:
    """
    Fit ICA on raw data and return the ICA object.
    """
    print(f"  Running ICA (n_components={cfg.N_ICA_COMPONENTS}, rs={random_state})...")
    raw_for_ica = raw.copy().filter(1.0, None, picks='eeg', fir_window='hamming')

    ica = mne.preprocessing.ICA(
        n_components=cfg.N_ICA_COMPONENTS,
        method='fastica',
        random_state=random_state,
        max_iter=1000,
    )
    ica.fit(raw_for_ica, picks='eeg')
    print("  ICA fit OK")
    return ica


def apply_basic_preprocessing(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Apply basic preprocessing: montage, filter, resample.
    """
    print("  Setting montage...")
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='warn')

    if raw.info['sfreq'] != cfg.SFREQ_TARGET:
        print(f"  Resampling: {raw.info['sfreq']} → {cfg.SFREQ_TARGET} Hz")
        raw.resample(cfg.SFREQ_TARGET, npad='auto')

    print(f"  Filtering: notch {cfg.NOTCH_FREQ} Hz, bandpass {cfg.L_FREQ}–{cfg.H_FREQ} Hz")
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


# --- Main browser functions ---

def browse_raw(participant_id: str, block: str, preprocess: bool = True):
    """
    Load and browse raw EEG data interactively.
    """
    print(f"\n{'='*60}")
    print(f"  BROWSING RAW DATA: {participant_id}, Block {block}")
    print(f"{'='*60}\n")

    try:
        raw = load_raw_participant_block(participant_id, block)
    except Exception as e:
        print(f"  ✗ Error loading data: {e}")
        return

    if preprocess:
        raw = apply_basic_preprocessing(raw)

    print(f"\n  Data info:")
    print(f"    Channels: {raw.info['nchan']}")
    print(f"    Duration: {raw.times[-1]:.1f} s")
    print(f"    Sampling: {raw.info['sfreq']} Hz")

    events, event_id = mne.events_from_annotations(raw)
    print(f"    Events:   {len(events)}")
    for name, code in event_id.items():
        count = np.sum(events[:, 2] == code)
        print(f"      {name}: {count}")

    print("\n  Opening raw browser...")
    print("  Keyboard shortcuts:")
    print("    'b' – mark bad segment")
    print("    'a' – add annotation")
    print("    'z' – zoom")
    print("    'up/down' – change scale")
    print("    'q' – close")

    raw.plot(
        n_channels=20,
        scalings='auto',
        title=f"{participant_id} Block {block} – Raw Data",
        block=True,
        show=True,
        duration=30,
        start=0,
    )


def browse_epochs(participant_id: str, block: str, condition: str = 'dev'):
    """
    Load and browse preprocessed epochs interactively.
    """
    print(f"\n{'='*60}")
    print(f"  BROWSING EPOCHS: {participant_id}, Block {block}, {condition}")
    print(f"{'='*60}\n")

    epochs = load_epochs_participant_block(participant_id, block, condition)
    if epochs is None:
        return

    print(f"\n  Epochs info:")
    print(f"    N epochs: {len(epochs)}")
    print(f"    Time:     {epochs.tmin:.1f} – {epochs.tmax:.1f} s")
    print(f"    Channels: {len(epochs.ch_names)}")

    print("\n  Opening epochs browser...")
    print("  Click on an epoch to mark it as bad.")
    print("  Press 'd' to drop all marked epochs.")
    print("  Keyboard shortcuts: 'up/down' – scroll, 'q' – close")

    epochs.plot(
        n_epochs=20,
        n_channels=15,
        scalings='auto',
        title=f"{participant_id} Block {block} – {condition} epochs",
        show=True,
        block=True,
    )


def browse_ica(participant_id: str, block: str, random_state: int = cfg.RANDOM_STATE):
    """
    Run ICA on raw data and browse components interactively.
    """
    print(f"\n{'='*60}")
    print(f"  BROWSING ICA: {participant_id}, Block {block}")
    print(f"{'='*60}\n")

    try:
        raw = load_raw_participant_block(participant_id, block)
    except Exception as e:
        print(f"  ✗ Error loading data: {e}")
        return

    raw = apply_basic_preprocessing(raw)

    # Try ICA with given random state, fallback to 123 if fails
    try:
        ica = load_ica_from_raw(raw, random_state)
    except Exception as e:
        print(f"  ICA failed with rs={random_state}: {e}")
        print("  Retrying with random_state=123, max_iter=2000...")
        try:
            raw_for_ica = raw.copy().filter(1.0, None, picks='eeg')
            ica = mne.preprocessing.ICA(
                n_components=cfg.N_ICA_COMPONENTS,
                method='fastica',
                random_state=123,
                max_iter=2000,
            )
            ica.fit(raw_for_ica, picks='eeg')
            print("  ICA (fallback) OK")
        except Exception as e2:
            print(f"  ✗ ICA fallback also failed: {e2}")
            return

    print("\n  Opening ICA component browser...")
    print("  Click on a component to toggle selection.")
    print("  Components selected for removal will be highlighted.")
    print("  The plot shows topographies of each component.")
    print("  Close the plot to continue.")

    ica.plot_components(show=True)

    print("\n  Opening ICA sources browser (first 15 components)...")
    print("  This shows the time course of each ICA component.")
    print("  Look for components that show eye blinks, heartbeats, or muscle artifacts.")

    raw_for_ica = raw.copy().filter(1.0, None, picks='eeg')
    ica.plot_sources(
        raw_for_ica,
        picks=list(range(min(15, ica.n_components_))),
        show=True,
        start=0,
        stop=30,
        title=f"{participant_id} Block {block} – ICA Sources",
    )

    print("\n  Looking for EOG components...")
    try:
        eog_proxy = next((ch for ch in ['Fp1', 'Fp2'] if ch in raw.ch_names), None)
        if eog_proxy:
            eog_indices, scores = ica.find_bads_eog(raw, ch_name=eog_proxy, threshold=3.0)
            print(f"  Found EOG-like components: {[int(i) for i in eog_indices]}")
        else:
            print("  No Fp1/Fp2 found – skipping EOG detection.")
    except Exception as e:
        print(f"  EOG detection error: {e}")

    try:
        # Look for heartbeat via frontal channels
        ecg_channels = [ch for ch in ['ECG', 'EKG', 'Fp1', 'Fp2'] if ch in raw.ch_names]
        if ecg_channels:
            ecg_indices, scores = ica.find_bads_ecg(raw, ch_name=ecg_channels[0], threshold='auto')
            print(f"  Found ECG-like components: {[int(i) for i in ecg_indices]}")
    except Exception as e:
        print(f"  ECG detection skipped: {e}")

    # Apply ICA and show cleaned data
    print("\n  Applying ICA to remove selected components...")
    print("  (Components selected in the topomap plot will be removed)")
    print("  To remove components, click on them in the topomap plot, then close it.")

    # Actually, let's not apply automatically - let the user decide
    print("\n  If you want to see the effect of removing components, use:")
    print("    raw_clean = ica.apply(raw.copy(), exclude=[list_of_component_indices])")
    print("    raw_clean.plot()")


def browse_psd(participant_id: str, block: str):
    """
    Plot power spectral density of raw data.
    """
    print(f"\n{'='*60}")
    print(f"  POWER SPECTRUM: {participant_id}, Block {block}")
    print(f"{'='*60}\n")

    try:
        raw = load_raw_participant_block(participant_id, block)
    except Exception as e:
        print(f"  ✗ Error loading data: {e}")
        return

    raw = apply_basic_preprocessing(raw)

    print("  Plotting power spectrum...")
    raw.plot_psd(
        fmin=0.1,
        fmax=50,
        picks='eeg',
        average=True,
        show=True,
        title=f"{participant_id} Block {block} – Power Spectrum",
    )

    print("  Plotting power spectrum topomap (1–30 Hz)...")
    raw.plot_psd_topo(
        fmin=1,
        fmax=30,
        show=True,
        title=f"{participant_id} Block {block} – PSD Topomap",
    )


def browse_comparison(participant_id: str, block: str):
    """
    Load both standard and deviant epochs for comparison.
    """
    print(f"\n{'='*60}")
    print(f"  COMPARISON: {participant_id}, Block {block}")
    print(f"{'='*60}\n")

    epochs_std = load_epochs_participant_block(participant_id, block, 'std')
    epochs_dev = load_epochs_participant_block(participant_id, block, 'dev')

    if epochs_std is None or epochs_dev is None:
        print("  Cannot load both conditions.")
        return

    evoked_std = epochs_std.average()
    evoked_dev = epochs_dev.average()

    print("  Plotting standard vs deviant ERPs...")
    mne.viz.plot_evoked_overlay(
        [evoked_std, evoked_dev],
        picks=['Fz', 'Cz'],
        show=True,
        title=f"{participant_id} Block {block} – Standard vs Deviant",
    )

    print("  Plotting difference wave (deviant - standard)...")
    evoked_diff = mne.combine_evoked([evoked_dev, evoked_std], weights=[1, -1])
    evoked_diff.plot(
        picks=['Fz', 'Cz'],
        show=True,
        title=f"{participant_id} Block {block} – Difference (Dev - Std)",
    )

    evoked_diff.plot_joint(
        picks=['Fz'],
        show=True,
        title=f"{participant_id} Block {block} – MMN at Fz",
    )


def browse_topo(participant_id: str, block: str):
    """
    Plot topographical maps of the deviant ERP at MMN and P3a windows.
    """
    print(f"\n{'='*60}")
    print(f"  TOPOGRAPHY: {participant_id}, Block {block}")
    print(f"{'='*60}\n")

    epochs_dev = load_epochs_participant_block(participant_id, block, 'dev')
    if epochs_dev is None:
        return

    evoked = epochs_dev.average()

    t_mmn = (cfg.MMN_TMIN + cfg.MMN_TMAX) / 2
    print(f"  Topomap at MMN window ({cfg.MMN_TMIN*1000:.0f}–{cfg.MMN_TMAX*1000:.0f} ms)...")
    evoked.plot_topomap(
        times=[t_mmn],
        average=(cfg.MMN_TMAX - cfg.MMN_TMIN),
        sphere=_sphere_covering_all_channels(evoked.info),
        extrapolate='head',
        show=True,
        title=f"{participant_id} Block {block} – Deviant ERP at {t_mmn*1000:.0f} ms",
    )

    t_p3a = (cfg.P3A_TMIN + cfg.P3A_TMAX) / 2
    print(f"  Topomap at P3a window ({cfg.P3A_TMIN*1000:.0f}–{cfg.P3A_TMAX*1000:.0f} ms)...")
    evoked.plot_topomap(
        times=[t_p3a],
        average=(cfg.P3A_TMAX - cfg.P3A_TMIN),
        sphere=_sphere_covering_all_channels(evoked.info),
        extrapolate='head',
        show=True,
        title=f"{participant_id} Block {block} – Deviant ERP at {t_p3a*1000:.0f} ms",
    )

    evoked.plot_joint(
        picks=['Fz', 'Cz', 'Pz'],
        show=True,
        title=f"{participant_id} Block {block} – ERP with Topomaps",
    )


def browse_grand_average():
    """
    Load all participants' epochs and compute grand average for visual inspection.
    """
    print(f"\n{'='*60}")
    print("  GRAND AVERAGE BROWSER")
    print(f"{'='*60}\n")

    participants = cfg.PARTICIPANTS

    for block in ['A', 'B']:
        print(f"\n  Block {block}:")

        evokeds = []
        for p in participants:
            epochs = load_epochs_participant_block(p, block, 'dev')
            if epochs is not None and len(epochs) > 10:
                evokeds.append(epochs.average())

        if len(evokeds) < 5:
            print(f"    Not enough participants ({len(evokeds)}) for grand average.")
            continue

        ga = mne.grand_average(evokeds)
        print(f"    Grand average from {len(evokeds)} participants")

        ga.plot(
            picks=['Fz', 'Cz'],
            show=True,
            title=f"Grand Average – Block {block} Deviant (N={len(evokeds)})",
        )

        t_mmn = (cfg.MMN_TMIN + cfg.MMN_TMAX) / 2
        ga.plot_topomap(
            times=[t_mmn],
            average=(cfg.MMN_TMAX - cfg.MMN_TMIN),
            sphere=_sphere_covering_all_channels(ga.info),
            extrapolate='head',
            show=True,
            title=f"GA Block {block} – MMN at {t_mmn*1000:.0f} ms",
        )


def browse_annotations(participant_id: str, block: str):
    """
    Load raw data and show annotations (events) on the timeline.
    """
    print(f"\n{'='*60}")
    print(f"  ANNOTATIONS: {participant_id}, Block {block}")
    print(f"{'='*60}\n")

    try:
        raw = load_raw_participant_block(participant_id, block)
    except Exception as e:
        print(f"  ✗ Error loading data: {e}")
        return

    raw = apply_basic_preprocessing(raw)

    events, event_id = mne.events_from_annotations(raw)

    print("\n  Events found:")
    for name, code in sorted(event_id.items(), key=lambda x: x[1]):
        count = np.sum(events[:, 2] == code)
        print(f"    {name}: {count}")

    raw.plot(
        n_channels=20,
        scalings='auto',
        title=f"{participant_id} Block {block} – Annotations",
        show=True,
        block=True,
        duration=60,
    )


# --- MAIN ---

def main():
    parser = argparse.ArgumentParser(
        description="Interactive EEG browser for manual artifact inspection."
    )
    parser.add_argument(
        '--participant', '-p',
        type=str,
        help="Participant ID (e.g., P01)"
    )
    parser.add_argument(
        '--block', '-b',
        type=str,
        choices=['A', 'B', 'C'],
        help="Block letter (A, B, or C)"
    )
    parser.add_argument(
        '--condition', '-c',
        type=str,
        default='dev',
        choices=['std', 'dev'],
        help="Condition for epochs browsing (std or dev, default: dev)"
    )
    parser.add_argument(
        '--raw-only',
        action='store_true',
        help="Load raw data without preprocessing (only montage)"
    )
    parser.add_argument(
        '--epochs-only',
        action='store_true',
        help="Skip raw loading, load preprocessed epochs directly"
    )
    parser.add_argument(
        '--ica',
        action='store_true',
        help="Run ICA and browse components"
    )
    parser.add_argument(
        '--ica-rs',
        type=int,
        default=cfg.RANDOM_STATE,
        help="Random state for ICA (default: from config)"
    )
    parser.add_argument(
        '--psd',
        action='store_true',
        help="Plot power spectral density"
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help="Compare standard vs deviant ERPs"
    )
    parser.add_argument(
        '--topo',
        action='store_true',
        help="Plot topographical maps"
    )
    parser.add_argument(
        '--ga',
        action='store_true',
        help="Compute and browse grand average across all participants"
    )
    parser.add_argument(
        '--annotations',
        action='store_true',
        help="Show annotations/events on the raw data timeline"
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help="Run all browsing modes sequentially for the given participant/block"
    )

    args = parser.parse_args()

    # --- Grand average mode (doesn't need participant/block) ---
    if args.ga:
        browse_grand_average()
        return

    # --- Validation ---
    if not args.participant or not args.block:
        print("Error: --participant and --block are required (except for --ga).")
        print("Example: python analysis/manual_eeg_browser.py --participant P01 --block A")
        parser.print_help()
        sys.exit(1)

    participant = args.participant
    block = args.block

    # --- All mode ---
    if args.all:
        print(f"\n{'#'*60}")
        print(f"  RUNNING ALL BROWSING MODES: {participant}, Block {block}")
        print(f"{'#'*60}")

        browse_raw(participant, block, preprocess=not args.raw_only)

        browse_epochs(participant, block, 'dev')

        browse_epochs(participant, block, 'std')

        browse_ica(participant, block, args.ica_rs)

        browse_psd(participant, block)

        browse_comparison(participant, block)

        browse_topo(participant, block)

        browse_annotations(participant, block)

        print(f"\n{'#'*60}")
        print("  ALL MODES COMPLETED")
        print(f"{'#'*60}")
        return

    # --- Individual modes ---
    if args.epochs_only:
        browse_epochs(participant, block, args.condition)
        return

    if args.ica:
        browse_ica(participant, block, args.ica_rs)
        return

    if args.psd:
        browse_psd(participant, block)
        return

    if args.compare:
        browse_comparison(participant, block)
        return

    if args.topo:
        browse_topo(participant, block)
        return

    if args.annotations:
        browse_annotations(participant, block)
        return

    # Default: raw browser
    browse_raw(participant, block, preprocess=not args.raw_only)


if __name__ == '__main__':
    main()