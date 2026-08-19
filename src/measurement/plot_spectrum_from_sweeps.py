"""
Computes and compares power spectra for WFS sweep recordings at 2 m and 5 m
virtual source distances. Uses files from measurement_scenes/recorded_RIR/.

Beyond the raw spectra (where the level difference is dominated by the
inverse-square law), the script also computes spectra normalized against a
reference band below the aliasing frequency, in order to assess whether the
SHAPE of the spectrum (rather than just its level) is symmetric above the
spatial aliasing frequency.
"""

import numpy as np
import soundfile as sf
from scipy.signal import welch
import matplotlib.pyplot as plt
import glob
import os

# ============================================================================
# CONFIGURATION
# ============================================================================
FS = 48000                      # sample rate of the recordings
F_ALIAS = 974.4                 # theoretical spatial aliasing frequency [Hz]
NPERSEG = 4096                  # Welch segment length
NOVERLAP = 2048                 # segment overlap (50%)

# Reference band used for normalization (below aliasing, away from the band
# edges, where recording SNR is good). This is used to estimate the constant
# level offset between the 2 m and 5 m conditions caused by source distance.
REF_BAND_LOW = 100.0             # Hz
REF_BAND_HIGH = 500.0            # Hz

# Paths to the recordings (glob patterns)
REC_DIR = "data/measurement_scenes/recorded_RIR"
PATTERN_2M = os.path.join(REC_DIR, "2m-*.wav")
PATTERN_5M = os.path.join(REC_DIR, "5m-*.wav")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_and_average_spectrum(file_pattern, label):
    """
    Loads all files matching the pattern, computes the power spectral density
    (PSD) for each of them, and returns the averaged spectrum together with
    the corresponding frequency vector.
    """
    files = sorted(glob.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No files found for {label}: {file_pattern}")

    print(f"\n{label}: found {len(files)} files:")
    for f in files:
        print(f"  - {os.path.basename(f)}")

    psd_list = []
    f_axis = None

    for filepath in files:
        data, fs = sf.read(filepath)
        if fs != FS:
            print(f"  Warning: {os.path.basename(filepath)} has fs={fs}, expected {FS}. Using {fs}.")
        if data.ndim > 1:
            data = data.mean(axis=1)   # average channels (mono)

        # Compute PSD using Welch's method
        f, Pxx = welch(data, fs, nperseg=NPERSEG, noverlap=NOVERLAP)
        Pxx_db = 10 * np.log10(Pxx + 1e-12)   # convert to dB

        psd_list.append(Pxx_db)
        f_axis = f

    # Average the spectra across all files
    avg_psd = np.mean(psd_list, axis=0)
    return f_axis, avg_psd


def reference_band_offset(f, psd, low, high):
    """
    Returns the median PSD [dB] within the given reference band (used to
    normalize out the constant level offset between distances).
    """
    mask = (f >= low) & (f <= high)
    if not np.any(mask):
        raise ValueError(f"Reference band {low}-{high} Hz is empty on this frequency grid.")
    return np.median(psd[mask])

# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    print("=" * 60)
    print("SPECTRUM ANALYSIS FOR WFS SWEEPS (2 m vs 5 m)")
    print("=" * 60)

    # 1. Compute averaged spectra for both distances
    f_2m, psd_2m = load_and_average_spectrum(PATTERN_2M, "2 m")
    f_5m, psd_5m = load_and_average_spectrum(PATTERN_5M, "5 m")

    if not np.allclose(f_2m, f_5m):
        raise ValueError(
            "Frequency vectors for 2 m and 5 m differ — check that both sets of "
            "recordings share the same sample rate and Welch segment length."
        )
    f = f_2m

    # 2. Normalization: remove the constant level offset (distance -> 1/r
    #    law) estimated from the reference band, to compare spectral SHAPE.
    offset_2m = reference_band_offset(f, psd_2m, REF_BAND_LOW, REF_BAND_HIGH)
    offset_5m = reference_band_offset(f, psd_5m, REF_BAND_LOW, REF_BAND_HIGH)
    level_offset_db = offset_2m - offset_5m

    psd_2m_norm = psd_2m - offset_2m
    psd_5m_norm = psd_5m - offset_5m

    print(f"\nLevel offset (2 m - 5 m) in the reference band "
          f"{REF_BAND_LOW:.0f}-{REF_BAND_HIGH:.0f} Hz: {level_offset_db:.2f} dB")
    print("(This is mainly the effect of the inverse-square law for source "
          "distance, and is not relevant to the assessment of aliasing symmetry.)")

    # 3a. Figure 1: raw spectra
    fig1, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.semilogx(f, psd_2m, label='2 m (average of 3 recordings)', linewidth=1.8, color='blue')
    ax1.semilogx(f, psd_5m, label='5 m (average of 3 recordings)', linewidth=1.8, color='orange')
    ax1.axvline(x=F_ALIAS, color='red', linestyle='--', linewidth=1.5,
                label=f'$f_{{alias}}$ = {F_ALIAS:.1f} Hz')
    y_min = min(np.min(psd_2m), np.min(psd_5m))
    y_max = max(np.max(psd_2m), np.max(psd_5m))
    ax1.set_ylim(y_min - 5, y_max + 5)
    ax1.set_xlabel('Frequency [Hz]', fontsize=11)
    ax1.set_ylabel('PSD [dB]', fontsize=11)
    # No title — this is a standalone figure now, description goes in the
    # LaTeX \caption{} ("raw spectra, level difference ~ 1/r law").
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(loc='best', fontsize=9)
    ax1.set_xlim(20, 20000)
    fig1.tight_layout()

    out_path_raw = "data/measurement_scenes/spectrum_2m_vs_5m_raw.png"
    fig1.savefig(out_path_raw, dpi=150, bbox_inches='tight')
    print(f"\n✅ Plot saved as: {out_path_raw}")

    # 3b. Figure 2: spectra normalized against the reference band
    fig2, ax2 = plt.subplots(figsize=(10, 4.5))
    ax2.semilogx(f, psd_2m_norm, label='2 m (normalized)', linewidth=1.8, color='blue')
    ax2.semilogx(f, psd_5m_norm, label='5 m (normalized)', linewidth=1.8, color='orange')
    ax2.axvline(x=F_ALIAS, color='red', linestyle='--', linewidth=1.5,
                label=f'$f_{{alias}}$ = {F_ALIAS:.1f} Hz')
    ax2.axvspan(REF_BAND_LOW, REF_BAND_HIGH, color='gray', alpha=0.15,
                label=f'reference band ({REF_BAND_LOW:.0f}-{REF_BAND_HIGH:.0f} Hz)')
    y_min_n = min(np.min(psd_2m_norm), np.min(psd_5m_norm))
    y_max_n = max(np.max(psd_2m_norm), np.max(psd_5m_norm))
    ax2.set_ylim(y_min_n - 5, y_max_n + 5)
    ax2.set_xlabel('Frequency [Hz]', fontsize=11)
    ax2.set_ylabel('Normalized PSD [dB]', fontsize=11)
    # No title — description ("normalized to a common level, shape
    # comparison") goes in the LaTeX \caption{}.
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(loc='best', fontsize=9)
    ax2.set_xlim(20, 20000)
    fig2.tight_layout()

    out_path_norm = "data/measurement_scenes/spectrum_2m_vs_5m_normalized.png"
    fig2.savefig(out_path_norm, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved as: {out_path_norm}")

    plt.show()

    # 4. Numerical analysis: focus on the SHAPE above aliasing, after normalization
    above_mask = f >= F_ALIAS
    diff_raw_above = psd_2m[above_mask] - psd_5m[above_mask]
    diff_norm_above = psd_2m_norm[above_mask] - psd_5m_norm[above_mask]

    print("\n--- Spectral differences above f_alias ({:.1f} Hz) ---".format(F_ALIAS))
    print(f"Raw level difference:            mean = {np.mean(diff_raw_above):.2f} dB, "
          f"std = {np.std(diff_raw_above):.2f} dB")
    print(f"Difference AFTER shape normalization: mean = {np.mean(diff_norm_above):.2f} dB, "
          f"std = {np.std(diff_norm_above):.2f} dB")

    idx_alias = np.argmin(np.abs(f - F_ALIAS))
    print(f"\nAt f = {f[idx_alias]:.0f} Hz (normalized):")
    print(f"  2 m: {psd_2m_norm[idx_alias]:.2f} dB")
    print(f"  5 m: {psd_5m_norm[idx_alias]:.2f} dB")
    print(f"  Difference: {psd_2m_norm[idx_alias] - psd_5m_norm[idx_alias]:.2f} dB")

    max_diff_idx = np.argmax(np.abs(diff_norm_above))
    f_above = f[above_mask]
    print(f"\nMaximum SHAPE difference (after normalization) above f_alias: "
          f"{diff_norm_above[max_diff_idx]:.2f} dB at {f_above[max_diff_idx]:.0f} Hz")

# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    main()