import numpy as np
import soundfile as sf
from pathlib import Path


# Hardware configuration
N_PER_WALL = 15
N_WALLS = 4
N_CHANNELS = N_PER_WALL * N_WALLS  # Total of 60 speakers
FS = 48000


# Diagnostic parameters
NOISE_DURATION = 0.5   # duration of noise per speaker in seconds
SILENCE_DURATION = 0.1 # silence before the next speaker in seconds
NOISE_LEVEL = 0.05     # noise amplitude (low, to avoid SPL surprises at start)


OUTPUT_FILE = Path('wfs_speaker_diagnostic.wav')


def generate_white_noise(duration_sec, fs, level):
    """Generates white noise with given amplitude and duration."""
    samples = int(duration_sec * fs)
    return np.random.normal(0, level, samples)


def create_diagnostic_file():
    noise_samples = int(NOISE_DURATION * FS)
    silence_samples = int(SILENCE_DURATION * FS)
    step_samples = noise_samples + silence_samples
    total_samples = N_CHANNELS * step_samples


    # Initialize multi-channel matrix (samples, channels)
    # soundfile expects shape where columns are channels
    audio_mix = np.zeros((total_samples, N_CHANNELS))


    print(f"Generating test sequence for {N_CHANNELS} channels...")


    for ch in range(N_CHANNELS):
        start_idx = ch * step_samples
        end_idx = start_idx + noise_samples
       
        # Filling a single channel with noise
        audio_mix[start_idx:end_idx, ch] = generate_white_noise(NOISE_DURATION, FS, NOISE_LEVEL)


    # Saving to file
    sf.write(OUTPUT_FILE, audio_mix, FS)
    print(f"  Saved diagnostic file: {OUTPUT_FILE}")
    print(f"  Duration: {total_samples / FS:.1f} seconds")
    print(f"  Format: {N_CHANNELS}-channel WAV file, {FS} Hz")


if __name__ == '__main__':
    create_diagnostic_file()