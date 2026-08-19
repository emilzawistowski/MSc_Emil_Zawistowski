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

    data_all, fs = sf.read(WAV_PATH)   # (n_samples, n_channels), because sf.write(audio_mix.T, ...)
    print(f"File: {data_all.shape[0]} samples, {data_all.shape[1]} channels, fs={fs}\n")

    # NOTE: We do NOT sort globally by t_target_s. This log contains events
    # from 7 separate, repeated playback sessions of the same (deterministic)
    # calib_combined.wav file - in each session t_target_s resets to the same
    # pattern (1..19 s for 2 m, 24..42 s for 5 m). Global sorting by t_target_s
    # would mix rows from different sessions and destroy the boundaries between them.
    log = pd.read_csv(CSV_PATH).reset_index(drop=True)

    distances_sorted = sorted(meta['distances'].items(), key=lambda kv: kv[1]['trigger_code'])

    all_delays_ms = []
    for dist_str, info in distances_sorted:
        trig_code = info['trigger_code']
        ref_channel = info['reference_channel']

        channel_data = data_all[:, ref_channel]

        subset = log[log['trigger_code'] == trig_code].reset_index(drop=True)

        # The log contains 7 repeated playback sessions of the same WAV file.
        # We detect session boundaries by a drop in t_target_s (return to the beginning of the pattern),
        # and then in EACH session individually we skip the first and last trial
        # (t_target might be close to 0 or the sequence start has different scheduling dynamics).
        t = subset['t_target_s'].values
        run_id = np.zeros(len(t), dtype=int)
        run_counter = 0
        for i in range(1, len(t)):
            if t[i] < t[i - 1] - 0.5:  # value dropped -> new session
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
            print(f"Distance {dist_str} m: NO onsets detected on channel {ref_channel} "
                  f"- check if this channel actually contains anything (might be silent/zero).")
            continue

        delays_ms = np.array(delays_ms)
        mean_ms = delays_ms.mean()
        std_ms  = delays_ms.std()

        print(f"Distance {dist_str} m (ref channel={ref_channel}, n={len(delays_ms)}/{len(subset_mid)}):")
        print(f"  mean delay in file (onset - t_target) = {mean_ms:.3f} ms "
              f"(std={std_ms:.3f} ms)\n")

        all_delays_ms.append(delays_ms)

    if len(all_delays_ms) == 2:
        pooled = np.concatenate(all_delays_ms)
        print("=== RESULT ===")
        print(f"L_sys (mean of offsets, both distances, n={len(pooled)}): "
              f"{pooled.mean():.3f} ms (std={pooled.std():.3f} ms)")
        print("\nNOTE: this method does NOT measure real acoustics (air), only the time")
        print("from sending the trigger to the moment the given sample appears in the WAV file")
        print("played by Reaper - i.e., the Python->Reaper->ASIO buffer chain.")
        print("This does NOT include the real DAC->air delay (fraction of a ms,")
        print("negligible) nor any potential unforeseen RME driver delays -")
        print("but it is the only measurement possible to make without additional hardware.")


if __name__ == '__main__':
    main()