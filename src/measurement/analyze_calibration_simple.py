import numpy as np
import pandas as pd
import soundfile as sf
import json

WAV_PATH      = 'rendered_blocks/calib_combined.wav'
CSV_PATH      = 'events_log_calib_combined.csv'
METADATA_PATH = 'rendered_blocks/calib_combined_metadata.json'


def detect_onset(data, fs, expected_sample, window_s=0.03, rel_threshold=0.2):
    half = int(window_s * fs)
    start = max(0, expected_sample - half)
    end   = min(len(data), expected_sample + half)
    segment = data[start:end]
    if len(segment) == 0 or np.max(np.abs(segment)) == 0:
        return None
    threshold = rel_threshold * np.max(np.abs(segment))
    above = np.abs(segment) > threshold
    idx = np.argmax(above)
    if not above[idx]:
        return None
    return start + idx


def main():
    with open(METADATA_PATH) as f:
        meta = json.load(f)

    data_all, fs = sf.read(WAV_PATH)                                                                


    log = pd.read_csv(CSV_PATH).reset_index(drop=True)

    distances_sorted = sorted(meta['distances'].items(), key=lambda kv: kv[1]['trigger_code'])

    all_delays_ms = []
    for dist_str, info in distances_sorted:
        trig_code = info['trigger_code']
        ref_channel = info['reference_channel']

        channel_data = data_all[:, ref_channel]

        subset = log[log['trigger_code'] == trig_code].reset_index(drop=True)


        t = subset['t_target_s'].values
        run_id = np.zeros(len(t), dtype=int)
        run_counter = 0
        for i in range(1, len(t)):
            if t[i] < t[i - 1] - 0.5:                                
                run_counter += 1
            run_id[i] = run_counter
        subset = subset.copy()
        subset['run_id'] = run_id

        subset_mid = subset.groupby('run_id', group_keys=False).apply(
            lambda g: g.iloc[1:-1] if len(g) > 2 else g
        )

        delays_ms = []
        for _, row in subset_mid.iterrows():
            t_target = row['t_target_s']
            expected_sample = int(t_target * fs)
            onset_sample = detect_onset(channel_data, fs, expected_sample)
            if onset_sample is not None:
                t_onset = onset_sample / fs
                delays_ms.append((t_onset - t_target) * 1000)

        if not delays_ms:


            continue

        delays_ms = np.array(delays_ms)
        mean_ms = delays_ms.mean()
        std_ms  = delays_ms.std()


        all_delays_ms.append(delays_ms)

    if len(all_delays_ms) == 2:
        pooled = np.concatenate(all_delays_ms)


if __name__ == '__main__':
    main()