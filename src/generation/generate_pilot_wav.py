import numpy as np
import soundfile as sf
import csv
import random
from pathlib import Path
from tqdm import tqdm


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


N_TRIALS = 50
ISI_SECONDS = 0.5                                        
RESPONSE_PAUSE = 3.0                                              

DISTANCE_NEAR = 2.0
DISTANCE_FAR = 5.0

TOKEN_INDEX = 1                                       


def generate_silence(duration_sec, n_speakers):
    """Generates silence of given duration."""
    n_samples = int(duration_sec * FS)
    return np.zeros((n_speakers, n_samples))

def render_single_token(distance, target_wav, array):
    """Renders a single token at the specified distance."""
    source_pos = azimuth_to_cartesian(TARGET_AZIMUTH, distance)
    return render_point_source_wfs(target_wav, FS, source_pos, array)


def main():


    array = build_square_array(N_PER_WALL, SPACING, WALL_DIST)
    n_speakers = len(array.x)


    target_files = sorted(TARGET_DIR.glob('*.wav'))
    if len(target_files) < TOKEN_INDEX + 1:
        raise ValueError(f"Token with index {TOKEN_INDEX} not found")

    wav_path = target_files[TOKEN_INDEX]
    target_wav = load_and_verify_audio(wav_path, target_duration=TARGET_DURATION)
    target_wav = apply_spl_calibration(target_wav, TARGET_SPL)


    random.seed(42)                       
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


    full_mix = np.empty((n_speakers, 0))
    silence_isi = generate_silence(ISI_SECONDS, n_speakers)
    silence_pause = generate_silence(RESPONSE_PAUSE, n_speakers)

    for trial in tqdm(trials, desc="Trials"):

        sig1 = render_single_token(trial['distance_1'], target_wav, array)

        sig2 = render_single_token(trial['distance_2'], target_wav, array)


        trial_audio = np.concatenate([sig1, silence_isi, sig2, silence_pause], axis=1)


        full_mix = np.concatenate([full_mix, trial_audio], axis=1)


    output_wav = Path("pilot_2afc_audio.wav")

    sf.write(output_wav, full_mix.T, FS)


    output_log = Path("pilot_2afc_log.csv")

    with open(output_log, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['trial', 'distance_1', 'distance_2', 'correct_response'])
        writer.writeheader()
        for trial in trials:

            row = {k: v for k, v in trial.items() if k != 'near_first'}
            writer.writerow(row)


if __name__ == "__main__":
    main()