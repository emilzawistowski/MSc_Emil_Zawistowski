from pathlib import Path

# ---------------------------------------------------------------------------
# 1. PATHS
# ---------------------------------------------------------------------------
ROOT_PATH = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT_PATH / 'data'
RAW_EEG_PATH = DATA_PATH / 'raw_eeg'
EEG_DATA_PATH = RAW_EEG_PATH / 'eeg_data'
VHDR_DATA_PATH = RAW_EEG_PATH / 'vhdr_data'
VMRK_DATA_PATH = RAW_EEG_PATH / 'vmrk_data'
PREPROCESSED_PATH = DATA_PATH / 'processed_eeg'
TMP_PATH = PREPROCESSED_PATH / 'tmp'
VMRK_CLEANED_PATH = PREPROCESSED_PATH / 'vmrk_cleaned'

STIMULI_PATH = DATA_PATH / 'stimuli_selected'
MASKER_STIMULI_PATH = STIMULI_PATH / 'masker_stimuli'
TARGETS_READY_PATH = STIMULI_PATH / 'targets_ready'

MEASUREMENT_SCENES_PATH = DATA_PATH / 'measurement_scenes'
RENDERED_BLOCKS_PATH = DATA_PATH / 'rendered_blocks'
JITTER_PATH = DATA_PATH / 'jitter'
PILOT_PATH = DATA_PATH / 'pilot'

RESULTS_PATH = ROOT_PATH / 'results'
FIGURES_PATH = RESULTS_PATH / 'figures'
STATS_PATH = RESULTS_PATH / 'tables'

FIGURES_ERP_PATH = FIGURES_PATH / 'erp'
FIGURES_TOPO_PATH = FIGURES_PATH / 'topomap'
FIGURES_DECODING_PATH = FIGURES_PATH / 'decoding'
FIGURES_INDIVIDUAL_PATH = FIGURES_PATH / 'individual_erps'

REPORTS_PATH = RESULTS_PATH / 'reports'
SRC_PATH = ROOT_PATH / 'src'

OUTPUT_DIRS = [
    PREPROCESSED_PATH,
    TMP_PATH,
    STIMULI_PATH,
    MASKER_STIMULI_PATH,
    TARGETS_READY_PATH,
    MEASUREMENT_SCENES_PATH,
    RENDERED_BLOCKS_PATH,
    JITTER_PATH,
    PILOT_PATH,
    FIGURES_ERP_PATH,
    FIGURES_TOPO_PATH,
    FIGURES_DECODING_PATH,
    FIGURES_INDIVIDUAL_PATH,
    REPORTS_PATH,
    STATS_PATH,
]

# ---------------------------------------------------------------------------
# 2. PARTICIPANTS
# ---------------------------------------------------------------------------

PARTICIPANTS = [
    'P01', 'P02', 'P03', 'P04', 'P05',
    'P06', 'P07', 'P08', 'P09', 'P10',
    'P11', 'P12', 'P13', 'P14', 'P15',
    'P16', 'P17', 'P18', 'P19', 'P20'
]

BLOCK_LETTER_TO_TYPE = {'a': 'A', 'b': 'B', 'c': 'C'}

# ---------------------------------------------------------------------------
# 3. ACQUISITION AND PREPROCESSING PARAMETERS
# ---------------------------------------------------------------------------
SFREQ_TARGET = 500
L_FREQ = 0.1
H_FREQ = 40
NOTCH_FREQ = 50.0

EPOCH_TMIN = -0.2
EPOCH_TMAX = 0.5     # Kept as is
BASELINE = (-0.2, 0)

REJECT_THRESHOLD = 100e-6
MIN_DEVIANT_TRIALS = 40

N_ICA_COMPONENTS = 25
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 3a. ICA vs. bad-channel interpolation ordering
# ---------------------------------------------------------------------------
# Interpolating bad channels before fitting ICA reduces the effective rank
# of the data, which caused rank-deficiency warnings in several blocks
# (mainly ones with many interpolated channels). Fitting ICA on the good
# channels only, then interpolating afterwards, avoids this and is the
# standard fix recommended in the EEG literature.
#
# Options: 'ica_before_interpolate' (used here), 'interpolate_before_ica'
# (other approach), 'reduce_ica_components' (alternative fix, shrinks
# N_ICA_COMPONENTS per block instead of reordering the steps).
ICA_STRATEGY = 'ica_before_interpolate'

MIN_ICA_COMPONENTS = 8  

LEAD_IN = 15

# Whether to remove standards before deviant
EXCLUDE_PRE_DEVIANT_STANDARDS = True

# Bad channel detection
BAD_CH_THRESHOLD_STD = 2.5  
BAD_CH_CORRELATION_THRESHOLD = 0.4  # low correlation with neighbors

# ---------------------------------------------------------------------------
# 4. TIME WINDOWS, ROI, AND TRIGGER OFFSETS
# ---------------------------------------------------------------------------
MMN_TMIN = 0.150
MMN_TMAX = 0.300
MMN_LATENCY_TMIN = 0.100
MMN_LATENCY_TMAX = 0.280
P3A_TMIN = 0.250
P3A_TMAX = 0.500
SSA_TMIN = 0.150
SSA_TMAX = 0.300

ROI_MMN = ['Fz', 'Cz', 'FC1', 'FC2']
ROI_P3A = ['Cz', 'Pz', 'FC1', 'FC2']
ROI_EXTRA = ['T7', 'T8']
ALL_ROI_CHANNELS = list(dict.fromkeys(ROI_MMN + ROI_P3A + ROI_EXTRA))

SYSTEM_DELAY = 0.000208
C = 343.0

FLIGHT_TIME = {
    2.0: 2.0 / C,
    2.8: 2.8 / C,
    3.0: 3.0 / C,
    3.4: 3.4 / C,
    4.0: 4.0 / C,
    5.0: 5.0 / C,
}

TRIGGER_OFFSETS = {
    11: SYSTEM_DELAY + FLIGHT_TIME[5.0],
    12: SYSTEM_DELAY + FLIGHT_TIME[2.0],
    21: SYSTEM_DELAY + FLIGHT_TIME[2.0],
    22: SYSTEM_DELAY + FLIGHT_TIME[5.0],
    31: SYSTEM_DELAY + FLIGHT_TIME[3.4],
    99: 0.0,
}

# ---------------------------------------------------------------------------
# 5. TRIGGER CODES
# ---------------------------------------------------------------------------
TRIGGER_CODES = {
    'A_standard': 'S 11',
    'A_deviant': 'S 12',
    'B_standard': 'S 21',
    'B_deviant': 'S 22',
    'C_control': 'S 31',
    'block_start': 'S 99',
}

TRIGGER_CODES_INT = {
    'A_standard': 11,
    'A_deviant': 12,
    'B_standard': 21,
    'B_deviant': 22,
    'C_control': 31,
    'block_start': 99,
}

# ---------------------------------------------------------------------------
# 6a. Automatic exclusion of bad recordings (00a_raw_noise_check.py)
# ---------------------------------------------------------------------------
# Thresholds used by 00b_flag_bad_recordings.py to hard-exclude broken
# recordings before preprocessing. Two recordings (P04-B, P18-C) showed
# clear signs of a broken/saturated channel; thresholds below are set
# conservatively so normal between-participant variability isn't flagged.

RISING_NOISE_RMS_RATIO_THRESHOLD = 2.0   # rms_last_vs_first_ratio; catches rising noise over the recording
SIGNAL_POWER_ROBUST_Z_THRESHOLD = 5.0    # MAD-based |z| on log10 signal power; catches saturation/outlier channels
FLAT_RMS_REL_STD_THRESHOLD = 1e-6        # rel. std of RMS across last 3 quarters; near-zero indicates clipping

EXCLUDED_RECORDINGS_PATH = STATS_PATH / 'excluded_recordings.csv'

# ---------------------------------------------------------------------------
# 6. STATISTICAL / DECODING ANALYSIS PARAMETERS
# ---------------------------------------------------------------------------
N_CSP_COMPONENTS = 6
N_PERMUTATIONS = 10_000
BF10_THRESHOLD = 3.0
DECODING_THRESHOLD = 0.55

# ---------------------------------------------------------------------------
# 7. HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def get_participant_files(participant_id: str) -> dict:
    result = {}
    for block_letter in ['a', 'b', 'c']:
        block_type = block_letter.upper()
        pattern = f"{participant_id}_*_{block_letter}"
        vhdr_files = list(VHDR_DATA_PATH.glob(f"{pattern}.vhdr"))
        eeg_files = list(EEG_DATA_PATH.glob(f"{pattern}.eeg"))
        vmrk_cleaned_files = list(VMRK_CLEANED_PATH.glob(f"{pattern}.vmrk"))

        if not vhdr_files:
            print(f"  WARNING: Missing .vhdr file for {participant_id} block {block_letter.upper()}")
            continue

        result[block_type] = {
            'vhdr': vhdr_files[0],
            'eeg': eeg_files[0] if eeg_files else None,
            'vmrk_cleaned': vmrk_cleaned_files[0] if vmrk_cleaned_files else None,
        }
    return result

def ensure_output_dirs() -> None:
    for path in OUTPUT_DIRS:
        path.mkdir(parents=True, exist_ok=True)
    print("Output directories ready.")

# ---------------------------------------------------------------------------
# 7a. HARD EXCLUSION LIST (populated by 00b_flag_bad_recordings.py)
# ---------------------------------------------------------------------------
_EXCLUDED_RECORDINGS_CACHE = None  # module-level cache; loaded once per run

def load_excluded_recordings() -> frozenset:
    """Loads (participant, block) pairs flagged for exclusion by
    00b_flag_bad_recordings.py. Returns an empty set if that script
    hasn't been run yet, so downstream scripts don't crash."""
    global _EXCLUDED_RECORDINGS_CACHE
    if _EXCLUDED_RECORDINGS_CACHE is not None:
        return _EXCLUDED_RECORDINGS_CACHE

    if not EXCLUDED_RECORDINGS_PATH.exists():
        print(f"  WARNING: {EXCLUDED_RECORDINGS_PATH.name} not found - no recordings "
              f"will be excluded. Run 00a_raw_noise_check.py then "
              f"00b_flag_bad_recordings.py first to enable automatic exclusion.")
        _EXCLUDED_RECORDINGS_CACHE = frozenset()
        return _EXCLUDED_RECORDINGS_CACHE

    import csv
    pairs = set()
    with open(EXCLUDED_RECORDINGS_PATH, 'r', encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            pairs.add((row['participant'], row['block']))

    _EXCLUDED_RECORDINGS_CACHE = frozenset(pairs)
    return _EXCLUDED_RECORDINGS_CACHE

def is_excluded(participant_id: str, block: str) -> bool:
    """True if this participant/block was hard-excluded by
    00b_flag_bad_recordings.py (see raw_noise_check.csv / excluded_recordings.csv)."""
    return (participant_id, block) in load_excluded_recordings()

# ---------------------------------------------------------------------------
# 8. QUICK TEST
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print(f"SYSTEM_DELAY = {SYSTEM_DELAY*1000:.3f} ms")
    print(f"LEAD_IN = {LEAD_IN}")
    print(f"EXCLUDE_PRE_DEVIANT_STANDARDS = {EXCLUDE_PRE_DEVIANT_STANDARDS}")
    for code, offset in sorted(TRIGGER_OFFSETS.items()):
        print(f"code {code}: {offset*1000:.3f} ms")