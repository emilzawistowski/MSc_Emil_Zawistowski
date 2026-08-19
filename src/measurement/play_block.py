"""
Possible arguments:
--block A
--block B
--block C
"""

import argparse
import serial
import time
import json
import requests
import statistics
import csv
import os

REAPER_BASE_URL = 'http://127.0.0.1:8082'
COM_PORT = 'COM4'
FS = 48000

ACTION_PLAY = '40073'
ACTION_STOP = '40044'


def send_trigger(ser, value):
    """
    Returns the exact time (perf_counter_ns) at the moment the trigger is sent.
    """
    try:
        ser.write(bytes([value & 0xFF]))
        ser.flush()  # important: forces the UART buffer to flush

        t_actual = time.perf_counter_ns()

        time.sleep(0.05)

        ser.write(bytes([0]))
        ser.flush()

        return t_actual

    except Exception as e:
        print(f"[ERROR] Trigger send failed {value}: {e}")
        return None


def run_block(metadata_path, out_csv="events_log.csv"):

    # --- JSON ---
    with open(metadata_path) as f:
        meta = json.load(f)

    trials = meta['trials']
    trigger_times = [t['onset_sample'] / FS for t in trials]
    trigger_values = [t['trigger_code'] for t in trials]

    print(f"[OK] Loaded {len(trials)} triggers.")

    # --- SERIAL ---
    ser = serial.Serial(COM_PORT, 115200, timeout=1)
    time.sleep(0.5)

    # --- REAPER CHECK ---
    requests.get(f"{REAPER_BASE_URL}/_/")

    input("\nPRESS ENTER TO START")

    requests.get(f"{REAPER_BASE_URL}/_/{ACTION_PLAY}")
    t_start = time.perf_counter_ns()

    print("[OK] STARTED")

    jitters = []
    results = []

    for i, (t_target, trig_val) in enumerate(zip(trigger_times, trigger_values), 1):

        t_send = t_start + int(t_target * 1e9)

        # pre-sleep
        while True:
            now = time.perf_counter_ns()
            if now >= t_send - 2_000_000:  # 2 ms safety margin
                break

        # spin wait (high precision)
        while time.perf_counter_ns() < t_send:
            pass

        t_actual = send_trigger(ser, trig_val)

        if t_actual is not None:
            jitter_ns = t_actual - t_send
            jitters.append(jitter_ns)

            print(
                f"[{i}/{len(trials)}] "
                f"code={trig_val} "
                f"jitter={jitter_ns/1e6:.3f} ms"
            )
        
        t_actual_s = (t_actual - t_start) / 1e9
        t_target_s = t_target

        jitter_ms = jitter_ns / 1e6

        results.append([
            i,
            trig_val,
            t_target_s,
            t_actual,
            t_actual_s,
            jitter_ms
        ])
    
    header = [
        "trial_index",
        "trigger_code",
        "t_target_s",
        "t_actual_ns",
        "t_actual_s",
        "jitter_ms"
    ]

    file_exists = os.path.isfile(out_csv)

    with open(out_csv, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(header)

        writer.writerows(results)

    print(f"\n[OK] Data saved to {out_csv}")

    # --- STOP ---
    ser.close()
    requests.get(f"{REAPER_BASE_URL}/_/{ACTION_STOP}")

    # --- JITTER STATISTICS ---
    if jitters:
        mean_jitter = statistics.mean(jitters) / 1e6
        std_jitter = statistics.pstdev(jitters) / 1e6
        max_jitter = max(jitters, key=abs) / 1e6

        print("\n=== JITTER STATS ===")
        print(f"Mean: {mean_jitter:.3f} ms")
        print(f"STD : {std_jitter:.3f} ms")
        print(f"Max : {max_jitter:.3f} ms")

    print("\nDone.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--block', choices=['A', 'B', 'C'], required=True)
    args = parser.parse_args()

    run_block(
        f'rendered_blocks/block_{args.block}_metadata.json',
        out_csv=f'events_log_{args.block}.csv'
    )