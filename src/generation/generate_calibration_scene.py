import numpy as np
import soundfile as sf
import json
import sfs
from pathlib import Path
from wfs_pipeline_scene_generator import (
    build_square_array, azimuth_to_cartesian,
    N_PER_WALL, SPACING, WALL_DIST, FS, TARGET_AZIMUTH, LISTENER_POS
)

OUTPUT_DIR = Path('./rendered_blocks')
OUTPUT_DIR.mkdir(exist_ok=True)

CAL_DISTANCES   = [2.0, 5.0]
N_REPEATS       = 10
SOA             = 2.0
CLICK_DUR       = 0.005
LEAD_SILENCE    = 1.0                                                        
GAP_BETWEEN     = 3.0                                                      
TRIGGER_CODES   = {2.0: 202, 5.0: 205}

def make_click(fs, duration=CLICK_DUR, freq=1000.0):
    n = int(duration * fs)
    t = np.arange(n) / fs
    ramp = np.minimum(t / (duration * 0.2), 1.0)
    return np.sin(2 * np.pi * freq * t) * ramp

def pick_reference_channel(source_pos, array):
    delays, weights, selection, _ = sfs.td.wfs.point_25d(
        array.x, array.n, source_pos, xref=LISTENER_POS
    )
    active_idx = np.where(selection.astype(bool))[0]
    if len(active_idx) == 0:
        raise RuntimeError(f"No active speakers for source_pos={source_pos}")
    ref_idx = active_idx[len(active_idx) // 2]
    return int(ref_idx), float(delays[ref_idx])

def render_single_channel_click(click, fs, source_pos, array, ref_idx):
    delays, weights, selection, _ = sfs.td.wfs.point_25d(
        array.x, array.n, source_pos, xref=LISTENER_POS
    )
    active = selection.astype(bool)
    tukey_weights = sfs.tapering.tukey(active, alpha=0.5)
    weights = weights * tukey_weights

    signal = (click, fs)
    d = sfs.td.wfs.driving_signals(delays, weights, signal)
    driving = np.asarray(d.data).T

    out = np.zeros_like(driving)
    out[ref_idx, :] = driving[ref_idx, :]
    return out

def main():
    array = build_square_array(N_PER_WALL, SPACING, WALL_DIST)
    click = make_click(FS)


    blocks = []
    cursor_time = LEAD_SILENCE
    metadata = {'fs': FS, 'soa_s': SOA, 'distances': {}, 'trials': []}

    for dist in CAL_DISTANCES:
        source_pos = azimuth_to_cartesian(TARGET_AZIMUTH, dist)
        ref_idx, theoretical_delay_s = pick_reference_channel(source_pos, array)


        metadata['distances'][str(dist)] = {
            'reference_channel': ref_idx,
            'reference_channel_wfs_delay_s': theoretical_delay_s,
            'trigger_code': TRIGGER_CODES[dist],
        }

        block_trials = []
        for i in range(N_REPEATS):
            onset_time = cursor_time + i * SOA
            block_trials.append({
                'trial_idx': i,
                'onset_time_s': onset_time,
                'onset_sample': int(onset_time * FS),
                'trigger_code': TRIGGER_CODES[dist],
                'distance_m': dist,
                'reference_channel': ref_idx,
            })
        blocks.append((dist, source_pos, ref_idx, block_trials))
        metadata['trials'].extend(block_trials)


        cursor_time = block_trials[-1]['onset_time_s'] + SOA + GAP_BETWEEN

    total_duration = cursor_time + 0.5                    
    total_samples = int(total_duration * FS)
    n_speakers = len(array.x)
    audio_mix = np.zeros((n_speakers, total_samples))


    for dist, source_pos, ref_idx, block_trials in blocks:
        for trial in block_trials:
            driving = render_single_channel_click(click, FS, source_pos, array, ref_idx)
            onset_sample = trial['onset_sample']
            end = onset_sample + driving.shape[1]
            if end <= total_samples:
                audio_mix[:, onset_sample:end] += driving

    wav_path  = OUTPUT_DIR / "calib_combined.wav"
    json_path = OUTPUT_DIR / "calib_combined_metadata.json"
    sf.write(wav_path, audio_mix.T, FS)
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)


if __name__ == '__main__':
    main()