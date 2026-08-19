import numpy as np
import soundfile as sf
import sfs
from sfs.array import linear as build_linear_wall, concatenate as concat_arrays
from pathlib import Path
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# TRIGGER CODES — centrally defined to avoid mistakes
# ============================================================================
TRIGGER_CODES = {
    # Block A: Far standard, Near deviant
    'A_standard':  11,
    'A_deviant':   12,
    # Block B: Near standard, Far deviant
    'B_standard':  21,
    'B_deviant':   22,
    # Block C: Many-standards control (one code for all)
    'C_control':   31,
    # Special codes
    'block_start': 99,
}

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================
N_PER_WALL = 15
SPACING    = 0.176
WALL_DIST  = 1.5

FS           = 48000
TARGET_SPL   = 66
MASKER_SPL   = 70

LISTENER_POS    = np.array([0.0, 0.0, 0.0])
TARGET_AZIMUTH  = 0

DISTANCE_NEAR         = 2.0
DISTANCE_FAR          = 5.0
DISTANCE_INTERMEDIATE = [2.8, 4.0]

MASKER_AZIMUTHS  = [60, -60, 120, -120]
MASKER_DISTANCE  = 3.0

SOA             = 0.900
TARGET_DURATION = 0.600

BLOCKS = {
    'A': {
        'name': 'Far_standard_Near_deviant',
        'standard_distance': DISTANCE_FAR,
        'deviant_distance':  DISTANCE_NEAR,
        'n_standards': 400,
        'n_deviants':  100,
        'lead_in': 15,
        'trigger_standard': TRIGGER_CODES['A_standard'],
        'trigger_deviant':  TRIGGER_CODES['A_deviant'],
    },
    'B': {
        'name': 'Near_standard_Far_deviant',
        'standard_distance': DISTANCE_NEAR,
        'deviant_distance':  DISTANCE_FAR,
        'n_standards': 400,
        'n_deviants':  100,
        'lead_in': 15,
        'trigger_standard': TRIGGER_CODES['B_standard'],
        'trigger_deviant':  TRIGGER_CODES['B_deviant'],
    },
    'C': {
        'name': 'Many_standards_control',
        'distances':     [DISTANCE_NEAR, 2.8, 4.0, DISTANCE_FAR, 3.4],
        'n_per_distance': 100,
        'lead_in': 15,
        'trigger_control': TRIGGER_CODES['C_control'],
    }
}

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg

TARGET_DIR  = cfg.TARGETS_READY_PATH
MASKER_DIR  = cfg.MASKER_STIMULI_PATH
OUTPUT_DIR  = cfg.RENDERED_BLOCKS_PATH
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


# ============================================================================
# ARRAY
# ============================================================================
def build_square_array(n_per_wall, spacing, wall_dist):
    half = (n_per_wall - 1) / 2 * spacing
    back  = build_linear_wall(n_per_wall, spacing, center=[0, -wall_dist, 0], orientation=[0, 1, 0])
    front = build_linear_wall(n_per_wall, spacing, center=[0,  wall_dist, 0], orientation=[0, -1, 0])
    right = build_linear_wall(n_per_wall, spacing, center=[ wall_dist, 0, 0], orientation=[-1, 0, 0])
    left  = build_linear_wall(n_per_wall, spacing, center=[-wall_dist, 0, 0], orientation=[ 1, 0, 0])
    return concat_arrays(left, front, right, back)


# ============================================================================
# HELPERS
# ============================================================================
def azimuth_to_cartesian(azimuth_deg, distance):
    azimuth_rad = np.radians(azimuth_deg)
    return np.array([distance * np.sin(azimuth_rad),
                     distance * np.cos(azimuth_rad),
                     0.0])

def render_point_source_wfs(audio, fs, source_position, array):
    delays, weights, selection, _ = sfs.td.wfs.point_25d(
        array.x, array.n, source_position, xref=LISTENER_POS
    )

    # --- Tukey tapering ---
    active = selection.astype(bool)
    tukey_weights = sfs.tapering.tukey(active, alpha=0.5)

    # Combine WFS weights and Tukey taper
    weights = weights * tukey_weights

    signal = (audio, fs)
    d = sfs.td.wfs.driving_signals(delays, weights, signal)

    driving = np.asarray(d.data).T  # (n_speakers, L)

    n_sp, n_out = driving.shape
    n_in = len(audio)

    if n_out > n_in:
        return driving[:, :n_in]

    out = np.zeros((n_sp, n_in))
    out[:, :n_out] = driving
    return out

def load_and_verify_audio(filepath, target_duration=None, target_fs=FS):
    audio, fs = sf.read(filepath)
    if fs != target_fs:
        from scipy.signal import resample
        audio = resample(audio, int(len(audio) * target_fs / fs))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if target_duration is not None:
        expected = int(target_duration * target_fs)
        if len(audio) != expected:
            print(f"  WARNING: {filepath.name}: {len(audio)} samples, expected {expected}")
    return audio

def apply_spl_calibration(audio, target_spl_db, reference_spl_db=94.0):
    rms = np.sqrt(np.mean(audio**2))
    if rms > 0:
        return audio * (10**((target_spl_db - reference_spl_db) / 20.0) / rms)
    return audio


# ============================================================================
# TRIAL SEQUENCE GENERATION
# ============================================================================
def generate_trial_sequence(block_config, block_type):
    """
    Returns a list of dictionaries describing each trial.
    Each trial contains:
      - type: 'standard' | 'deviant' | 'control'
      - distance: float (meters)
      - word_idx: int (CVC file index)
      - trigger_code: int (code to send to TriggerBox)
      - onset_sample: filled later in render_block
    """
    np.random.seed(42)
    sequence = []

    if block_type in ['A', 'B']:
        n_std   = block_config['n_standards']
        n_dev   = block_config['n_deviants']
        lead_in = block_config['lead_in']
        t_std   = block_config['trigger_standard']
        t_dev   = block_config['trigger_deviant']

        # Lead-in: standards only, to establish the sensory memory trace
        for _ in range(lead_in):
            sequence.append({
                'type':         'standard',
                'distance':     block_config['standard_distance'],
                'word_idx':     np.random.randint(0, 9),
                'trigger_code': t_std,
                'is_lead_in':   True,
            })

        remaining_std = n_std
        remaining_dev = n_dev
        while remaining_dev > 0:
            n_std_chunk = min(np.random.randint(2, 7), remaining_std)
            for _ in range(n_std_chunk):
                sequence.append({
                    'type':         'standard',
                    'distance':     block_config['standard_distance'],
                    'word_idx':     np.random.randint(0, 9),
                    'trigger_code': t_std,
                    'is_lead_in':   False,
                })
                remaining_std -= 1
            if remaining_dev > 0:
                sequence.append({
                    'type':         'deviant',
                    'distance':     block_config['deviant_distance'],
                    'word_idx':     np.random.randint(0, 9),
                    'trigger_code': t_dev,
                    'is_lead_in':   False,
                })
                remaining_dev -= 1

        while remaining_std > 0:
            sequence.append({
                'type':         'standard',
                'distance':     block_config['standard_distance'],
                'word_idx':     np.random.randint(0, 9),
                'trigger_code': t_std,
                'is_lead_in':   False,
            })
            remaining_std -= 1

    elif block_type == 'C':
        distances  = block_config['distances']
        n_per_dist = block_config['n_per_distance']
        lead_in    = block_config['lead_in']
        t_ctrl     = block_config['trigger_control']

        for _ in range(lead_in):
            sequence.append({
                'type':         'control',
                'distance':     float(np.random.choice(distances)),
                'word_idx':     np.random.randint(0, 9),
                'trigger_code': t_ctrl,
                'is_lead_in':   True,
            })

        all_trials = []
        for dist in distances:
            for _ in range(n_per_dist):
                all_trials.append({
                    'type':         'control',
                    'distance':     float(dist),
                    'word_idx':     np.random.randint(0, 9),
                    'trigger_code': t_ctrl,
                    'is_lead_in':   False,
                })
        np.random.shuffle(all_trials)
        sequence.extend(all_trials)

    return sequence


# ============================================================================
# BLOCK RENDERING
# ============================================================================
def render_block(block_type, block_config, target_wavs, masker_wavs, array):
    print(f"\n{'='*60}")
    print(f"Rendering Block {block_type}: {block_config['name']}")
    print(f"{'='*60}")

    trial_sequence = generate_trial_sequence(block_config, block_type)
    n_trials       = len(trial_sequence)
    total_samples  = int((n_trials * SOA + TARGET_DURATION) * FS)
    n_speakers     = len(array.x)
    audio_mix      = np.zeros((n_speakers, total_samples))

    # Metadata: general block info
    metadata = {
        'block_type':    block_type,
        'block_name':    block_config['name'],
        'fs':            FS,
        'soa_s':         SOA,
        'n_trials':      n_trials,
        'trigger_codes': {k: v for k, v in TRIGGER_CODES.items()},
        'trials': []
    }

    # --- Targets ---
    print(f"Rendering {n_trials} targets...")
    for trial_idx, trial in enumerate(tqdm(trial_sequence)):
        onset_time   = trial_idx * SOA
        onset_sample = int(onset_time * FS)
        onset_ms     = round(onset_time * 1000, 3)

        word_wav     = target_wavs[trial['word_idx']]
        source_pos   = azimuth_to_cartesian(TARGET_AZIMUTH, trial['distance'])
        driving_sigs = render_point_source_wfs(word_wav, FS, source_pos, array)

        end_sample = onset_sample + driving_sigs.shape[1]
        if end_sample <= total_samples:
            audio_mix[:, onset_sample:end_sample] += driving_sigs

        metadata['trials'].append({
            'trial_idx':    trial_idx,
            'onset_time_s': round(onset_time, 6),
            'onset_ms':     onset_ms,
            'onset_sample': onset_sample,          # <<< crucial for verification
            'type':         trial['type'],
            'distance_m':   trial['distance'],
            'word_idx':     trial['word_idx'],
            'trigger_code': trial['trigger_code'],  # <<< crucial
            'is_lead_in':   trial.get('is_lead_in', False),
        })

    # --- Maskers ---
    print(f"Rendering {len(MASKER_AZIMUTHS)} maskers...")
    for mask_idx, azimuth in enumerate(tqdm(MASKER_AZIMUTHS)):
        masker_wav    = masker_wavs[mask_idx % len(masker_wavs)]
        n_repeats     = int(np.ceil(total_samples / len(masker_wav)))
        masker_looped = np.tile(masker_wav, n_repeats)[:total_samples]
        masker_cal    = apply_spl_calibration(masker_looped, MASKER_SPL - 6)
        masker_pos    = azimuth_to_cartesian(azimuth, MASKER_DISTANCE)

        chunk_size = FS * 10
        for cs in range(0, len(masker_cal), chunk_size):
            ce    = min(cs + chunk_size, len(masker_cal))
            chunk = masker_cal[cs:ce]
            audio_mix[:, cs:ce] += render_point_source_wfs(chunk, FS, masker_pos, array)

    print(f"✓ Block {block_type} ready ({total_samples/FS:.1f} s, {n_trials} trials)")
    return audio_mix, metadata


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("="*60)
    print("WFS EXPERIMENT RENDERING PIPELINE v2")
    print("="*60)

    print("\n[1/4] Building WFS array...")
    array = build_square_array(N_PER_WALL, SPACING, WALL_DIST)
    print(f"✓ {len(array.x)} speakers")

    print("\n[2/4] Loading targets...")
    target_files = sorted(TARGET_DIR.glob('*.wav'))[:9]
    if len(target_files) != 9:
        raise ValueError(f"Need 9 CVC files, found {len(target_files)}")
    target_wavs = []
    for tf in target_files:
        wav = load_and_verify_audio(tf, target_duration=TARGET_DURATION)
        target_wavs.append(apply_spl_calibration(wav, TARGET_SPL))
        print(f"  ✓ {tf.name}")

    print("\n[3/4] Loading maskers...")
    masker_files = sorted(MASKER_DIR.glob('*.wav'))[:4]
    if len(masker_files) < 4:
        raise ValueError(f"Need 4 maskers, found {len(masker_files)}")
    masker_wavs = []
    for mf in masker_files:
        masker_wavs.append(load_and_verify_audio(mf))
        print(f"  ✓ {mf.name}")

    print("\n[4/4] Rendering blocks...")
    for block_type, block_config in BLOCKS.items():
        audio_mix, metadata = render_block(
            block_type, block_config, target_wavs, masker_wavs, array
        )
        wav_path  = OUTPUT_DIR / f"block_{block_type}_{block_config['name']}.wav"
        json_path = OUTPUT_DIR / f"block_{block_type}_metadata.json"

        sf.write(wav_path, audio_mix.T, FS)
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Self-check: print trigger summary
        codes = [t['trigger_code'] for t in metadata['trials']]
        from collections import Counter
        print(f"\n  Block {block_type} trigger summary:")
        for code, count in sorted(Counter(codes).items()):
            meaning = [k for k, v in TRIGGER_CODES.items() if v == code]
            print(f"    code {code:3d} ({meaning if meaning else '?'}): {count} times")
        print(f"  → {wav_path}")
        print(f"  → {json_path}")

    print("\n✓ DONE")

if __name__ == '__main__':
    main()