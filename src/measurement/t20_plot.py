"""
Reproduces the RT60 (T20) octave-band figure from raw REW .txt exports.

Input: rt60_meas1.txt, rt60_meas2.txt, rt60_meas3.txt
       (as exported by REW: Reverberation -> RT60 -> Export -> Text)

Metric: T20 (decay -5 dB to -25 dB), matching Table tab:rt60_app
        and the in-text values in Section 4.2.5 (mean 0.412 s, 500-2000 Hz).
"""

import numpy as np
import matplotlib.pyplot as plt

FILES = [
    "data/measurement_scenes/rt60/rt60_meas1.txt",
    "data/measurement_scenes/rt60/rt60_meas2.txt",
    "data/measurement_scenes/rt60/rt60_meas3.txt",
]


T20_COL = 4


BANDS = [125, 250, 500, 1000, 2000, 4000]


def parse_t20(path):
    freqs, t20s = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or not line[0].isdigit():
                continue
            parts = [p.strip() for p in line.split(",")]
            if parts[0] == "full":
                continue
            freq = float(parts[0])
            if freq not in BANDS:
                continue                                                 
            t20s.append(float(parts[T20_COL]))
            freqs.append(freq)
    return np.array(freqs), np.array(t20s)


def main():
    all_t20 = []
    freq_ref = None
    for fp in FILES:
        freqs, t20s = parse_t20(fp)
        order = np.argsort(freqs)
        freqs, t20s = freqs[order], t20s[order]
        if freq_ref is None:
            freq_ref = freqs
        all_t20.append(t20s)

    all_t20 = np.array(all_t20)                
    mean_t20 = all_t20.mean(axis=0)
    std_t20 = all_t20.std(axis=0, ddof=1)


    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    ax.errorbar(
        freq_ref, mean_t20, yerr=std_t20,
        fmt="o-", color="#1f4e8c", ecolor="#1f4e8c",
        elinewidth=1.2, capsize=4, markersize=6, linewidth=2,
    )

    ax.set_xscale("log")
    ax.set_xticks(BANDS)
    ax.set_xticklabels(["125", "250", "500", "1k", "2k", "4k"])
    ax.set_xlim(110, 4600)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("$T_{20}$ (s)")


    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    fig.savefig("data/measurement_scenes/rt60/rt60_decay.pdf")
    fig.savefig("data/measurement_scenes/rt60/rt60_decay.png", dpi=300)


if __name__ == "__main__":
    main()