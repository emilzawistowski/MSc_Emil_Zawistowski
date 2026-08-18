import numpy as np
import soundfile as sf
from pathlib import Path


N_PER_WALL = 15
N_WALLS = 4
N_CHANNELS = N_PER_WALL * N_WALLS                        
FS = 48000


NOISE_DURATION = 0.5                                             
SILENCE_DURATION = 0.1                                             
NOISE_LEVEL = 0.05                                                             


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


    audio_mix = np.zeros((total_samples, N_CHANNELS))


    for ch in range(N_CHANNELS):
        start_idx = ch * step_samples
        end_idx = start_idx + noise_samples


        audio_mix[start_idx:end_idx, ch] = generate_white_noise(NOISE_DURATION, FS, NOISE_LEVEL)


    sf.write(OUTPUT_FILE, audio_mix, FS)


if __name__ == '__main__':
    create_diagnostic_file()