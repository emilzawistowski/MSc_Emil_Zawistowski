import numpy as np
import soundfile as sf
import csv
import random
from pathlib import Path
from tqdm import tqdm

# Import your custom functions
from wfs_pipeline_scene_generator import (
    build_square_array,
    render_point_source_wfs,
    load_and_verify_audio,
    apply_spl_calibration,
    azimuth_to_cartesian,
    FS,
    TARGET_SPL,
    TARGET_AZIMUTH,
    TARGET_DURATION,
    N_PER_WALL,
    SPACING,
    WALL_DIST,
    TARGET_DIR,
    LISTENER_POS
)

# ==========================
# TRAINING CONFIGURATION
# ==========================
N_TRIALS = 10              # number of training trials
ISI_SECONDS = 0.5
RESPONSE_PAUSE = 3.0

DISTANCE_NEAR = 2.0
DISTANCE_FAR = 5.0

TOKEN_INDEX = 1  # same token as in the main experiment

# ==========================
# HELPER FUNCTIONS
# ==========================
def generate_silence(duration_sec, n_speakers):
    n_samples = int(duration_sec * FS)
    return np.zeros((n_speakers, n_samples))

def render_single_token(distance, target_wav, array):
    source_pos = azimuth_to_cartesian(TARGET_AZIMUTH, distance)
    return render_point_source_wfs(target_wav, FS, source_pos, array)

# ==========================
# MAIN FUNCTION
# ==========================
def main():
    print("="*60)
    print(f"Generating WAV file for TRAINING (2AFC, {N_TRIALS} trials)")
    print("="*60)

    # 1. Build the speaker array
    print("Building speaker array...")
    array = build_square_array(N_PER_WALL, SPACING, WALL_DIST)
    n_speakers = len(array.x)
    print(f"Number of speakers: {n_speakers}")

    # 2. Load one token
    print("Loading target token...")
    target_files = sorted(TARGET_DIR.glob('*.wav'))
    if len(target_files) < TOKEN_INDEX + 1:
        raise ValueError(f"Token with index {TOKEN_INDEX} not found")
    
    wav_path = target_files[TOKEN_INDEX]
    target_wav = load_and_verify_audio(wav_path, target_duration=TARGET_DURATION)
    target_wav = apply_spl_calibration(target_wav, TARGET_SPL)
    print(f"Loaded: {wav_path.name}")

    # 3. Prepare trial list (randomized order)
    print("Generating training trial sequence...")
    random.seed(42)  # same as main, but this need not be identical
    trials = []
    for trial_idx in range(1, N_TRIALS + 1):
        near_first = random.choice([True, False])
        if near_first:
            d1, d2 = DISTANCE_NEAR, DISTANCE_FAR
            correct = "1"
        else:
            d1, d2 = DISTANCE_FAR, DISTANCE_NEAR
            correct = "2"
        
        trials.append({
            'trial': trial_idx,
            'distance_1': d1,
            'distance_2': d2,
            'correct_response': correct,
            'near_first': near_first,
        })

    # 4. Render all trials and concatenate
    print("Rendering sounds and creating file...")
    full_mix = np.empty((n_speakers, 0))
    silence_isi = generate_silence(ISI_SECONDS, n_speakers)
    silence_pause = generate_silence(RESPONSE_PAUSE, n_speakers)

    for trial in tqdm(trials, desc="Training trials"):
        sig1 = render_single_token(trial['distance_1'], target_wav, array)
        sig2 = render_single_token(trial['distance_2'], target_wav, array)
        trial_audio = np.concatenate([sig1, silence_isi, sig2, silence_pause], axis=1)
        full_mix = np.concatenate([full_mix, trial_audio], axis=1)

    # 5. Save WAV file
    output_wav = Path("pilot_training_audio.wav")
    print(f"\nSaving audio file: {output_wav}")
    sf.write(output_wav, full_mix.T, FS)
    print(f"Saved: {output_wav} (duration: {full_mix.shape[1] / FS:.1f} s)")

    # 6. Save log (metadata) as CSV
    output_log = Path("pilot_training_log.csv")
    print(f"Saving log: {output_log}")
    with open(output_log, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['trial', 'distance_1', 'distance_2', 'correct_response'])
        writer.writeheader()
        writer.writerows(trials)
    
    print("\nDone!")
    print(f"1. Play the file {output_wav} in Reaper as TRAINING.")
    print("2. The participant does not need to record answers – this is just familiarization.")
    print(f"3. Then play the main pilot_2afc_audio.wav file and ask the participant to fill in the response sheet.")
    print("="*60)

if __name__ == "__main__":
    main()