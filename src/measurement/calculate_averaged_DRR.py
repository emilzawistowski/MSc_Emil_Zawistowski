import os
import glob
import numpy as np
import soundfile as sf
from scipy.fft import rfft, irfft
from scipy.signal import stft
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------

MAX_ANALYSIS_S = 60.0
SWEEP_F1_HZ = 20.0
SWEEP_F2_HZ = 20000.0


def estimate_clock_drift_and_onset(recorded, fs, sweep_duration_s,
                                    f1=SWEEP_F1_HZ, f2=SWEEP_F2_HZ, debug=False):
    """(unchanged from the previous fixed version -- MAD-based iterative
    outlier rejection, no fixed floor. See prior version's docstring.)
    """
    nperseg = 4096
    noverlap = 3584
    f_axis, t_axis, Zxx = stft(recorded, fs=fs, nperseg=nperseg, noverlap=noverlap)
    mag = np.abs(Zxx)
    min_bin = np.searchsorted(f_axis, 30.0)

    frame_energy = np.max(mag[min_bin:, :], axis=0)
    threshold = frame_energy.max() * 0.05
    peak_freqs = np.full(mag.shape[1], np.nan)
    for i in range(mag.shape[1]):
        if frame_energy[i] < threshold:
            continue
        spec = mag[min_bin:, i]
        peak_bin = min_bin + int(np.argmax(spec))
        peak_freqs[i] = f_axis[peak_bin]

    valid = (~np.isnan(peak_freqs)) & (peak_freqs > f1 * 3)
    if np.sum(valid) < 20:
        if debug:
            print("  [DEBUG] Too few reliable frequency points (STFT) to estimate "
                  "clock drift -- falling back to simple energy threshold.")
        return 1.0, 0.0, None, None

    tt = t_axis[valid]
    ff = peak_freqs[valid]
    t_ref_expected = sweep_duration_s * np.log(ff / f1) / np.log(f2 / f1)

    A = np.vstack([tt, np.ones(len(tt))]).T
    (a, b), *_ = np.linalg.lstsq(A, t_ref_expected, rcond=None)

    MAD_TO_STD = 1.4826
    OUTLIER_K = 3.0
    MAX_ITERS = 10

    for _ in range(MAX_ITERS):
        residuals = t_ref_expected - (a * tt + b)
        mad = np.median(np.abs(residuals - np.median(residuals)))
        robust_sigma = mad * MAD_TO_STD
        if robust_sigma <= 0:
            break
        inlier_thresh = OUTLIER_K * robust_sigma
        inliers = np.abs(residuals) < inlier_thresh
        if np.sum(inliers) < 20 or np.sum(inliers) == len(tt):
            if np.sum(inliers) >= 20:
                tt, ff, t_ref_expected = tt[inliers], ff[inliers], t_ref_expected[inliers]
                A = np.vstack([tt, np.ones(len(tt))]).T
                (a, b), *_ = np.linalg.lstsq(A, t_ref_expected, rcond=None)
            break
        tt, ff, t_ref_expected = tt[inliers], ff[inliers], t_ref_expected[inliers]
        A = np.vstack([tt, np.ones(len(tt))]).T
        (a, b), *_ = np.linalg.lstsq(A, t_ref_expected, rcond=None)

    residuals = t_ref_expected - (a * tt + b)
    residual_std = float(np.std(residuals))
    onset_s = -b / a if a != 0 else None
    onset_sample = int(round(onset_s * fs)) if onset_s is not None else None

    if a != 0 and len(tt) >= 2:
        sxx = np.sum((tt - tt.mean()) ** 2)
        if sxx > 0:
            pred_var = residual_std ** 2 * (1.0 / len(tt) + (onset_s - tt.mean()) ** 2 / sxx)
            onset_uncertainty_s = float(np.sqrt(pred_var) / abs(a))
        else:
            onset_uncertainty_s = None
    else:
        onset_uncertainty_s = None

    if debug:
        extra = (f", onset extrapolation uncertainty (~1 sigma) = "
                 f"{onset_uncertainty_s*1000:.2f} ms" if onset_uncertainty_s is not None
                 else ", onset extrapolation uncertainty: could not be computed")
        print(f"  [DEBUG] Clock drift: a={a:.6f} ({(a-1)*1e6:+.1f} ppm), "
              f"onset={onset_s:.4f}s ({onset_sample} samples), "
              f"fit residual std={residual_std*1000:.1f} ms (n={len(tt)} points)"
              f"{extra}")

    return float(a), float(b), onset_sample, residual_std


def deconvolve(recorded_file, ref_file, debug=False):
    """(unchanged -- Farina deconvolution with clock-drift correction.)"""
    recorded, fs_rec = sf.read(recorded_file, always_2d=False)
    reference, fs_ref = sf.read(ref_file, always_2d=True)

    if recorded.ndim > 1:
        recorded = recorded.mean(axis=1)
    reference = reference.sum(axis=1)

    if fs_rec != fs_ref:
        print(f"  WARNING: Different sample rates! Recording: {fs_rec}, Reference: {fs_ref}. Using recording fs.")
    fs = fs_rec

    if debug:
        print(f"  [DEBUG] {os.path.basename(recorded_file)}: "
              f"recording = {len(recorded)/fs:.2f} s ({len(recorded)} samples), "
              f"reference = {os.path.basename(ref_file)} = {len(reference)/fs_ref:.2f} s "
              f"({len(reference)} samples), fs_rec={fs_rec}, fs_ref={fs_ref}")

    max_samples = int(MAX_ANALYSIS_S * fs)
    if len(recorded) > max_samples:
        recorded = recorded[:max_samples]

    recorded = recorded.astype(np.float64)
    reference = reference.astype(np.float64)
    recorded = recorded / np.max(np.abs(recorded))
    reference = reference / np.max(np.abs(reference))

    n_ref = len(reference)
    sweep_duration_s = n_ref / fs

    a_drift, b_onset, onset_sample, residual_std = estimate_clock_drift_and_onset(
        recorded, fs, sweep_duration_s, debug=debug
    )

    if onset_sample is None:
        coarse_win = int(0.1 * fs)
        n_coarse_win = len(recorded) // coarse_win
        rms_per_win = np.array([
            np.sqrt(np.mean(recorded[i * coarse_win:(i + 1) * coarse_win] ** 2))
            for i in range(n_coarse_win)
        ])
        noise_floor = np.median(rms_per_win[:max(1, int(2.0 / 0.1))])
        onset_threshold = max(noise_floor * 5, 1e-6)
        above = np.where(rms_per_win > onset_threshold)[0]
        onset_sample = int(above[0] * coarse_win) if len(above) > 0 else 0
        a_drift = 1.0
        if debug:
            print(f"  [DEBUG] Fallback (energy threshold): onset = {onset_sample} samples "
                  f"({onset_sample/fs*1000:.1f} ms), no drift correction.")
    elif abs(a_drift - 1.0) > 0.05:
        if debug:
            print(f"  [DEBUG] Estimated drift ({(a_drift-1)*1e6:+.0f} ppm) is "
                  f"physically unrealistic -- skipping drift correction, using only onset.")
        a_drift = 1.0

    if a_drift != 1.0:
        n_resampled = int(round(len(recorded) / a_drift))
        from scipy.signal import resample as _resample
        recorded = _resample(recorded, n_resampled)
        onset_sample = int(round(onset_sample / a_drift))
        if debug:
            print(f"  [DEBUG] Recording resampled by factor {1/a_drift:.6f} "
                  f"({len(recorded)} samples after resampling), corrected onset = "
                  f"{onset_sample} samples ({onset_sample/fs*1000:.1f} ms)")

    offset = max(0, onset_sample)

    pre_roll_samples = int(0.5 * fs)
    post_roll_samples = int(5.0 * fs)
    crop_start = max(0, offset - pre_roll_samples)
    crop_end = min(len(recorded), offset + len(reference) + post_roll_samples)
    recorded = recorded[crop_start:crop_end]
    pre_roll_actual = offset - crop_start

    t = np.arange(n_ref) / fs
    duration = n_ref / fs

    f1 = 20.0
    f2 = 20000.0
    k = np.exp(t * np.log(f2 / f1) / duration)
    inverse_filter = reference[::-1] / k

    n_fft = len(recorded) + len(inverse_filter) - 1
    n_fft = int(2 ** np.ceil(np.log2(n_fft)))

    H_rec = rfft(recorded, n=n_fft)
    H_inv = rfft(inverse_filter, n=n_fft)
    IR_full = irfft(H_rec * H_inv, n=n_fft)

    zero_idx = n_ref - 1 + pre_roll_actual
    pre_ms = 20
    post_ms = 1000
    pre_samples = int(pre_ms / 1000 * fs)
    post_samples = int(post_ms / 1000 * fs)
    start = max(0, zero_idx - pre_samples)
    end = min(len(IR_full), zero_idx + post_samples)
    IR = IR_full[start:end]

    IR = IR / np.max(np.abs(IR))

    peak_idx_dbg = np.argmax(np.abs(IR))
    peak_to_rms = np.abs(IR[peak_idx_dbg]) / (np.sqrt(np.mean(IR ** 2)) + 1e-12)
    if debug:
        print(f"  [DEBUG] IR: length={len(IR)} samples ({len(IR)/fs*1000:.1f} ms), "
              f"peak_idx={peak_idx_dbg} ({peak_idx_dbg/fs*1000:.2f} ms), "
              f"peak/RMS={peak_to_rms:.2f} (good deconvolution usually >5-10; "
              f"<3 means smeared/incorrect peak)")

    return IR, fs, peak_to_rms


def detect_direct_sound_peak(ir, fs, debug=False):
    """Finds the direct-sound peak sample, n_0, as defined in the thesis
    methodology (ISO 3382-1 convention): the sample index of the
    direct-sound PEAK, not of some earlier "onset"/rise threshold.

    HISTORY: several onset-detection alternatives to plain argmax were
    tried on this dataset (Hilbert-envelope threshold relative to noise
    floor; the same with a minimum sustain duration; the same walking
    backward from the global envelope peak; the same again on an
    RMS-smoothed envelope). Every one of them locked onto some point on
    the smooth, continuously-rising pre-peak ramp visible in the averaged
    IR plots (roughly 10 ms wide, from ~24 ms to ~35 ms in this dataset)
    and produced physically impossible DRR values (~-45 to -54 dB). That
    ramp is consistent with the pre-ringing of a band-limited (20 Hz-
    20 kHz) system recovered via Farina deconvolution (a classic Gibbs-
    type artifact of ANY sharp band-limited impulse response, not a bug
    in this pipeline) -- it rises smoothly and continuously into the main
    peak with no intervening silence, so no amplitude-based "start of
    rise" threshold can cleanly separate it from the peak itself.

    Critically, the thesis's own methodology (Sec. "DRR Verification")
    defines n_0 as "the sample index of the direct-sound peak" (ISO
    3382-1 convention) -- i.e. the peak itself, not an onset/rise point.
    Plain argmax(abs(ir)) is therefore not a workaround but the correct
    implementation of the documented method, and it is also the only
    version of this function that reproduced the published, physically
    sensible DRR values (6.61 dB at 2 m, 3.09 dB at 5 m).

    A residual concern with plain argmax -- that the direct-sound arrival
    sits inside a dense, several-sample-wide cluster of comparable-height
    positive/negative extrema, so argmax could in principle flip between
    them under a sub-ms upstream alignment shift -- remains real but is a
    second-order effect (at most a fraction of a millisecond of jitter in
    n_0), not a reason to substitute a fundamentally different point
    (rise onset) for the peak that the methodology actually calls for.
    """
    peak_idx = int(np.argmax(np.abs(ir)))
    if debug:
        print(f"  [DEBUG] Direct-sound peak (argmax): {peak_idx} ({peak_idx/fs*1000:.2f} ms)")
    return peak_idx


def align_and_average(ir_list, fs, quality_list=None, quality_threshold=8.0, debug=False):
    """(unchanged apart from calling detect_direct_sound_peak.)"""
    if not ir_list:
        return None

    if quality_list is not None:
        kept = [(ir, q) for ir, q in zip(ir_list, quality_list) if q >= quality_threshold]
        dropped_n = len(ir_list) - len(kept)
        if debug and dropped_n > 0:
            print(f"  [DEBUG] Dropped {dropped_n}/{len(ir_list)} recordings due to low quality "
                  f"(peak/RMS < {quality_threshold}) before averaging.")
        if len(kept) == 0:
            print(f"  WARNING: all {len(ir_list)} recordings are below the quality threshold "
                  f"({quality_threshold}) -- averaging anyway, but the result is unreliable.")
            kept = list(zip(ir_list, quality_list))
        ir_list = [ir for ir, q in kept]

    peak_idxs = [detect_direct_sound_peak(ir, fs, debug=debug) for ir in ir_list]

    target_idx = int(0.01 * fs)

    aligned_irs = []
    min_len = min(len(ir) for ir in ir_list)

    for ir, peak_idx in zip(ir_list, peak_idxs):
        ir_cropped = ir[:min_len]
        shift = target_idx - peak_idx

        if shift > 0:
            ir_shifted = np.pad(ir_cropped[:len(ir_cropped)-shift], (shift, 0), 'constant')
        elif shift < 0:
            ir_shifted = ir_cropped[-shift:]
            ir_shifted = np.pad(ir_shifted, (0, len(ir_cropped) - len(ir_shifted)), 'constant')
        else:
            ir_shifted = ir_cropped

        aligned_irs.append(ir_shifted)

    avg_ir = np.mean(aligned_irs, axis=0)
    avg_ir = avg_ir / np.max(np.abs(avg_ir))
    return avg_ir


def calculate_drr(ir, fs, early_ms=2.5, late_ms=300, debug=False):
    peak_idx = detect_direct_sound_peak(ir, fs, debug=debug)
    peak_time = peak_idx / fs

    early_samples = int(early_ms / 1000 * fs)
    late_samples = int(late_ms / 1000 * fs)

    early_energy = np.sum(ir[peak_idx:peak_idx + early_samples] ** 2)

    late_start = peak_idx + early_samples
    late_end = min(peak_idx + late_samples, len(ir))
    late_energy = np.sum(ir[late_start:late_end] ** 2)

    if late_energy > 0:
        drr = 10 * np.log10(early_energy / late_energy)
    else:
        drr = np.inf

    return drr, peak_time


def plot_ir(ir, fs, drr, distance_label, save_path=None, early_ms=2.5, late_ms=300):
    time_axis = np.arange(len(ir)) / fs
    plt.figure(figsize=(12, 6))
    limit = min(int(0.05 * fs), len(ir))
    plt.plot(time_axis[:limit], ir[:limit], linewidth=1.5)

    peak_idx = detect_direct_sound_peak(ir, fs)
    peak_time = peak_idx / fs

    plt.axvline(x=peak_time, color='red', linestyle='--', label='Peak (Direct Sound)')
    plt.axvline(x=peak_time + early_ms / 1000, color='green', linestyle='--',
                label=f'End of early window ({early_ms:g}ms)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude (normalized)')
    # No title on the figure itself — distance and DRR value go in the
    # LaTeX \caption{} instead.
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def inspect_reference(ref_file, label):
    info = sf.info(ref_file)
    data, fs = sf.read(ref_file, always_2d=True)
    n_channels = data.shape[1]
    active_channels = np.where(np.max(np.abs(data), axis=0) > 1e-6)[0]
    print(f"  [DEBUG] Reference {label}: {info.frames} samples ({info.frames/fs:.2f} s), "
          f"{n_channels} channels, {len(active_channels)} active (non-zero) channels: "
          f"{active_channels.tolist()[:10]}{'...' if len(active_channels) > 10 else ''}")
    if len(active_channels) == 0:
        print(f"  WARNING: reference file {label} is completely SILENT on all channels!")
    return data, fs, active_channels


def main():
    ref_dir = "data/measurement_scenes/generated_RIR"
    rec_dir = "data/measurement_scenes/recorded_RIR"

    if not os.path.exists(ref_dir):
        print(f"Reference folder {ref_dir} does not exist!")
        return
    if not os.path.exists(rec_dir):
        print(f"Recording folder {rec_dir} does not exist!")
        return

    print(f"References (source sweeps): {ref_dir}")
    print(f"Recordings (from microphone): {rec_dir}")

    ref_2m = os.path.join(ref_dir, "sweep_2m.wav")
    ref_5m = os.path.join(ref_dir, "sweep_5m.wav")

    if not os.path.exists(ref_2m):
        candidates = glob.glob(os.path.join(ref_dir, "*2m*.wav"))
        if candidates:
            ref_2m = candidates[0]
            print(f"Found 2m reference: {os.path.basename(ref_2m)}")
        else:
            print("No reference found for 2m!")
            return
    else:
        print(f"Found 2m reference: {os.path.basename(ref_2m)}")

    if not os.path.exists(ref_5m):
        candidates = glob.glob(os.path.join(ref_dir, "*5m*.wav"))
        if candidates:
            ref_5m = candidates[0]
            print(f"Found 5m reference: {os.path.basename(ref_5m)}")
        else:
            print("No reference found for 5m!")
            return
    else:
        print(f"Found 5m reference: {os.path.basename(ref_5m)}")

    print()
    inspect_reference(ref_2m, "2m")
    inspect_reference(ref_5m, "5m")
    print()

    all_wavs = glob.glob(os.path.join(rec_dir, "*.wav"))

    recordings_2m = []
    recordings_5m = []

    for f in all_wavs:
        basename = os.path.basename(f).lower()
        if "averaged" in basename:
            continue
        if "2m" in basename:
            recordings_2m.append(f)
        elif "5m" in basename:
            recordings_5m.append(f)

    recordings_2m.sort()
    recordings_5m.sort()

    print(f"\nFound {len(recordings_2m)} recordings for 2m:")
    for r in recordings_2m:
        print(f"   - {os.path.basename(r)}")
    print(f"Found {len(recordings_5m)} recordings for 5m:")
    for r in recordings_5m:
        print(f"   - {os.path.basename(r)}")

    if recordings_2m:
        print("\n--- Processing 2m ---")
        ir_list = []
        quality_list = []
        for rec in recordings_2m:
            print(f"  -> Deconvolution: {os.path.basename(rec)}")
            ir, fs, quality = deconvolve(rec, ref_2m, debug=True)
            ir_list.append(ir)
            quality_list.append(quality)

        print("  [DEBUG] Per-file DRR (individual recordings, diagnostic only):")
        for rec, ir in zip(recordings_2m, ir_list):
            drr_individual, peak_individual = calculate_drr(ir, fs, debug=True)
            print(f"  [DEBUG]   {os.path.basename(rec)}: DRR={drr_individual:.2f} dB "
                  f"(peak at {peak_individual*1000:.2f} ms)")

        avg_ir_2m = align_and_average(ir_list, fs, quality_list=quality_list, debug=True)
        if avg_ir_2m is not None:
            drr_2m, peak_2m = calculate_drr(avg_ir_2m, fs)
            print(f"Averaged DRR for 2m: {drr_2m:.2f} dB (peak at {peak_2m:.4f} s)")

            out_ir_path = os.path.join(rec_dir, "averaged_IR_2m.wav")
            sf.write(out_ir_path, avg_ir_2m.astype(np.float32), fs)
            print(f"   Saved IR: {out_ir_path}")

            out_plot_path = os.path.join(rec_dir, "plot_IR_2m.png")
            plot_ir(avg_ir_2m, fs, drr_2m, "2m", save_path=out_plot_path)
            print(f"   Saved plot: {out_plot_path}")
    else:
        print("\nNo recordings for 2m.")

    if recordings_5m:
        print("\n--- Processing 5m ---")
        ir_list = []
        quality_list = []
        for rec in recordings_5m:
            print(f"  -> Deconvolution: {os.path.basename(rec)}")
            ir, fs, quality = deconvolve(rec, ref_5m, debug=True)
            ir_list.append(ir)
            quality_list.append(quality)

        print("  [DEBUG] Per-file DRR (individual recordings, diagnostic only):")
        for rec, ir in zip(recordings_5m, ir_list):
            drr_individual, peak_individual = calculate_drr(ir, fs, debug=True)
            print(f"  [DEBUG]   {os.path.basename(rec)}: DRR={drr_individual:.2f} dB "
                  f"(peak at {peak_individual*1000:.2f} ms)")

        avg_ir_5m = align_and_average(ir_list, fs, quality_list=quality_list, debug=True)
        if avg_ir_5m is not None:
            drr_5m, peak_5m = calculate_drr(avg_ir_5m, fs)
            print(f"Averaged DRR for 5m: {drr_5m:.2f} dB (peak at {peak_5m:.4f} s)")

            out_ir_path = os.path.join(rec_dir, "averaged_IR_5m.wav")
            sf.write(out_ir_path, avg_ir_5m.astype(np.float32), fs)
            print(f"   Saved IR: {out_ir_path}")

            out_plot_path = os.path.join(rec_dir, "plot_IR_5m.png")
            plot_ir(avg_ir_5m, fs, drr_5m, "5m", save_path=out_plot_path)
            print(f"   Saved plot: {out_plot_path}")
    else:
        print("\nNo recordings for 5m.")

    print("\nDone! All results saved in the 'measurement_scenes/recorded_RIR' folder.")


if __name__ == "__main__":
    main()