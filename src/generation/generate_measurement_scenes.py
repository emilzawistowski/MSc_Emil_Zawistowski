import argparse
import numpy as np
import soundfile as sf
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
    MASKER_SPL,
    MASKER_AZIMUTHS,
    MASKER_DISTANCE,
    TARGET_AZIMUTH,
    N_PER_WALL,
    SPACING,
    WALL_DIST,
    TARGET_DIR,
    MASKER_DIR,
    OUTPUT_DIR,
    TARGET_DURATION
)


MEASUREMENT_SOA = 1.5                                             
N_REPEATS = 20                                             
MASKER_DURATION = 30.0                                              


TOKEN_INDEX = 1                                 


SWEEP_DISTANCES = [2.0, 5.0]


def generate_maskers_only(array, masker_wavs):
    """Generates 30-second file with only maskers."""
    total_samples = int(MASKER_DURATION * FS)
    n_speakers = len(array.x)
    audio_mix = np.zeros((n_speakers, total_samples))


    for mask_idx, azimuth in enumerate(tqdm(MASKER_AZIMUTHS)):
        masker_wav = masker_wavs[mask_idx % len(masker_wavs)]

        n_repeats = int(np.ceil(total_samples / len(masker_wav)))
        masker_looped = np.tile(masker_wav, n_repeats)[:total_samples]
        masker_cal = apply_spl_calibration(masker_looped, MASKER_SPL - 6)                         
        masker_pos = azimuth_to_cartesian(azimuth, MASKER_DISTANCE)


        chunk_size = FS * 10
        for cs in range(0, len(masker_cal), chunk_size):
            ce = min(cs + chunk_size, len(masker_cal))
            chunk = masker_cal[cs:ce]
            audio_mix[:, cs:ce] += render_point_source_wfs(chunk, FS, masker_pos, array)

    return audio_mix


def generate_targets_only(array, target_wav, distance):
    """Generates file with repeated target at given distance."""

    total_duration = N_REPEATS * MEASUREMENT_SOA + 0.1                 
    total_samples = int(total_duration * FS)
    n_speakers = len(array.x)
    audio_mix = np.zeros((n_speakers, total_samples))


    source_pos = azimuth_to_cartesian(TARGET_AZIMUTH, distance)

    single_rendered = render_point_source_wfs(target_wav, FS, source_pos, array)


    for i in tqdm(range(N_REPEATS)):
        onset_sample = int(i * MEASUREMENT_SOA * FS)
        end_sample = onset_sample + single_rendered.shape[1]
        if end_sample <= total_samples:
            audio_mix[:, onset_sample:end_sample] += single_rendered

    return audio_mix


def load_sweep(sweep_path):
    """Loads the REW-generated sweep WAV and resamples/mixes to mono at FS if needed.

    The sweep is NOT SPL-calibrated here: absolute level for the measurement is
    set later via REW's own SPL calibration against the NoiSee reference
    (Section 4.3.6), so the sweep is only checked for sample rate / channel
    count and left at its native amplitude (expected to already be a safe
    playback level, e.g. -20 dBFS, as exported from REW).
    """
    wav, fs = sf.read(sweep_path)
    if fs != FS:
        from scipy.signal import resample
        wav = resample(wav, int(len(wav) * FS / fs))
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav.astype(np.float64)


def generate_sweep_at_distance(array, sweep_wav, distance):
    """Renders the calibration sweep as a WFS virtual point source at the given
    distance (same azimuth as the experimental targets), for later impulse-
    response extraction and DRR calculation in REW.

    Unlike generate_targets_only, this renders the sweep once, in full,
    without looping/repeating it: a single sweep is all that is needed for
    the deconvolution-based impulse response measurement.

    IMPORTANT: unlike generate_maskers_only, this renders the ENTIRE sweep
    in a single call to render_point_source_wfs, not in 10-second chunks.
    Chunking a continuous signal and rendering each chunk independently
    reintroduces a WFS delay-line discontinuity at every chunk boundary
    (each call effectively "restarts" the per-channel delay filter from
    zero, losing whatever tail/state would otherwise carry over from the
    previous chunk). This was confirmed to produce two audible "clicks" in
    the rendered sweep at t=10s and t=20s -- exactly the chunk boundaries
    used previously -- which a sine-sweep deconvolution then picks up as
    spurious extra "reflections" indistinguishable in the resulting IR
    plot from a true early room reflection. A 30 s sweep at 60 channels
    is a modest ~700 MB in memory as float64, well within reach of a single
    non-chunked render, so there is no need to trade correctness for the
    chunking that IS still needed (and fine) for the much longer,
    looped masker signal in generate_maskers_only above.
    """
    source_pos = azimuth_to_cartesian(TARGET_AZIMUTH, distance)


    audio_mix = render_point_source_wfs(sweep_wav, FS, source_pos, array)

    return audio_mix


def main():
    parser = argparse.ArgumentParser(description="Generate WFS measurement scenes (SPL calibration + DRR sweeps).")
    parser.add_argument(
        "--sweep",
        type=Path,
        default=Path("data/sweep/LogSwp.wav"),
        help="Path to a REW-generated sine sweep WAV file. If provided (or the "
             "default path exists), the sweep is additionally rendered as a WFS "
             "point source at each distance in SWEEP_DISTANCES, for impulse-"
             "response / DRR measurement.",
    )
    args = parser.parse_args()


    array = build_square_array(N_PER_WALL, SPACING, WALL_DIST)


    target_files = sorted(TARGET_DIR.glob('*.wav'))[:9]
    if len(target_files) < TOKEN_INDEX + 1:
        raise ValueError(f"Not enough target files, need at least {TOKEN_INDEX+1}")

    wav, fs = sf.read(target_files[TOKEN_INDEX])
    if fs != FS:
        from scipy.signal import resample
        wav = resample(wav, int(len(wav) * FS / fs))
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    expected_len = int(TARGET_DURATION * FS)
    if len(wav) > expected_len:
        wav = wav[:expected_len]
    elif len(wav) < expected_len:
        wav = np.pad(wav, (0, expected_len - len(wav)))
    target_wav = apply_spl_calibration(wav, TARGET_SPL)


    masker_files = sorted(MASKER_DIR.glob('*.wav'))[:4]
    if len(masker_files) < 4:
        raise ValueError(f"Need 4 maskers, found {len(masker_files)}")
    masker_wavs = []
    for mf in masker_files:
        masker_wavs.append(load_and_verify_audio(mf))


    output_dir = Path("./measurement_scenes")
    output_dir.mkdir(exist_ok=True)


    masker_mix = generate_maskers_only(array, masker_wavs)
    sf.write(output_dir / "maskers_only.wav", masker_mix.T, FS)


    mix_2m = generate_targets_only(array, target_wav, 2.0)
    sf.write(output_dir / "target_2m_only.wav", mix_2m.T, FS)


    mix_5m = generate_targets_only(array, target_wav, 5.0)
    sf.write(output_dir / "target_5m_only.wav", mix_5m.T, FS)


    if args.sweep is not None and args.sweep.exists():

        sweep_wav = load_sweep(args.sweep)

        for distance in SWEEP_DISTANCES:

            sweep_mix = generate_sweep_at_distance(array, sweep_wav, distance)
            out_name = f"sweep_{distance:g}m.wav"
            sf.write(output_dir / out_name, sweep_mix.T, FS)

    else:


        pass
if __name__ == "__main__":
    main()