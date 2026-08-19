import os
import re
import pandas as pd
import numpy as np

# --- YOUR FILE NAMES ---
files = [
    'rt60/rt60_meas1.txt',
    'rt60/rt60_meas2.txt',
    'rt60/rt60_meas3.txt'
]

# Bands of interest (125–4000 Hz)
target_freqs = [125, 250, 500, 1000, 2000, 4000]

# --- FUNCTION: EXTRACT T20 FROM FILE ---
def extract_t20(filename, target_freqs):
    """
    Reads a file exported from RT60 data as text and returns a dictionary:
    {freq: T20_value}
    """
    t20_dict = {}
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Look for data lines (starting with a number)
        if re.match(r'^\d+', line):
            parts = line.split(',')
            if len(parts) >= 5:
                try:
                    freq = float(parts[0].strip())
                    # INDEX 4 = T20 (not 5!)
                    t20 = float(parts[4].strip())
                    if freq in target_freqs:
                        t20_dict[freq] = t20
                except ValueError:
                    continue
    return t20_dict

# --- LOAD DATA FROM 3 FILES ---
all_data = []
for f in files:
    if os.path.exists(f):
        data = extract_t20(f, target_freqs)
        all_data.append(data)
        print(f"Loaded: {f}")
    else:
        print(f"File does not exist: {f}")

if not all_data:
    print("No data – check file names.")
    exit()

# --- CREATE TABLE ---
rows = []
for freq in target_freqs:
    values = [data.get(freq, np.nan) for data in all_data]
    if any(np.isnan(v) for v in values):
        print(f"Missing value for band {freq} Hz in some files.")
    mean_val = np.nanmean(values)
    std_val = np.nanstd(values)
    rows.append({
        'Frequency [Hz]': freq,
        'T20_1 [s]': values[0] if len(values) > 0 else np.nan,
        'T20_2 [s]': values[1] if len(values) > 1 else np.nan,
        'T20_3 [s]': values[2] if len(values) > 2 else np.nan,
        'Mean T20 [s]': round(mean_val, 3),
        'Std [s]': round(std_val, 3)
    })

df = pd.DataFrame(rows)

# --- DISPLAY ---
print("\n" + "="*70)
print("REVERBERATION TIME (T20) – AVERAGED FROM 3 MEASUREMENTS")
print("="*70)
print(df.to_string(index=False))
print("="*70)

# --- SAVE TO CSV ---
df.to_csv('rt60_t20_averaged.csv', index=False)
print("\nSaved to 'rt60_t20_averaged.csv'")

# --- AVERAGE FOR 500–2000 Hz BAND ---
mid_freqs = [500, 1000, 2000]
mid_values = df[df['Frequency [Hz]'].isin(mid_freqs)]['Mean T20 [s]'].values
if len(mid_values) == 3:
    avg_mid = round(np.mean(mid_values), 3)
    print(f"\nAverage T20 for 500–2000 Hz: {avg_mid} s")