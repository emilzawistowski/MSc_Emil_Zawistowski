import os
import soundfile as sf
import numpy as np
from scipy.signal import butter, lfilter


def process_stimuli(duration_ms=600, apply_filter=False, cutoff_hz=1000):
    input_folder = 'data/stimuli_selected/target_stimuli'
    output_folder = 'data/stimuli_selected/targets_ready'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = [f for f in os.listdir(input_folder) if f.lower().endswith('.wav')]

    for filename in files:
        path = os.path.join(input_folder, filename)
        out_path = os.path.join(output_folder, filename)

        try:
            data, sr = sf.read(path)

            # =========================================================
            # 1. Truncating + padding with silence
            # =========================================================
            n_samples = int((duration_ms / 1000.0) * sr)

            # truncate if too long
            trimmed_data = data[:n_samples]

            # add silence if too short
            if len(trimmed_data) < n_samples:
                pad_length = n_samples - len(trimmed_data)

                # mono
                if trimmed_data.ndim == 1:
                    trimmed_data = np.pad(
                        trimmed_data,
                        (0, pad_length),
                        mode='constant'
                    )

                # stereo / multi-channel
                else:
                    trimmed_data = np.pad(
                        trimmed_data,
                        ((0, pad_length), (0, 0)),
                        mode='constant'
                    )

            # =========================================================
            # 2. Filtering (Low-pass)
            # =========================================================
            if apply_filter:
                nyquist = 0.5 * sr
                normal_cutoff = cutoff_hz / nyquist

                # Butterworth filter
                b, a = butter(
                    N=2,
                    Wn=normal_cutoff,
                    btype='low',
                    analog=False
                )

                # filtering
                trimmed_data = lfilter(b, a, trimmed_data, axis=0)

            # =========================================================
            # 3. Fade-out (5 ms)
            # =========================================================
            fade_len = int(0.005 * sr)

            if len(trimmed_data) > fade_len:
                fade_curve = np.linspace(1.0, 0.0, fade_len)

                # mono
                if trimmed_data.ndim == 1:
                    trimmed_data[-fade_len:] *= fade_curve

                # stereo / multi-channel
                else:
                    trimmed_data[-fade_len:] *= fade_curve[:, np.newaxis]

            # =========================================================
            # 3B. Fade-in (10 ms) - Protection against N100 click artifact
            # =========================================================
            if len(trimmed_data) > 2 * fade_len:
                # Curve rising from 0.0 to 1.0
                fade_in_curve = np.linspace(0.0, 1.0, 2 * fade_len)
                
                # Mono
                if trimmed_data.ndim == 1:
                    trimmed_data[:fade_len * 2] *= fade_in_curve
                # Stereo / Multi-channel
                else:
                    trimmed_data[:fade_len * 2] *= fade_in_curve[:, np.newaxis]

            # =========================================================
            # 4. Save
            # =========================================================
            sf.write(out_path, trimmed_data, sr)

            filter_status = (
                f"LP {cutoff_hz}Hz"
                if apply_filter
                else "No filter"
            )

            print(f"OK: {filename} -> {duration_ms}ms | {filter_status}")

        except Exception as e:
            print(f"Error in {filename}: {e}")


if __name__ == "__main__":

    # Parameters:
    # duration_ms = target duration
    # apply_filter = True/False
    # cutoff_hz = LPF cutoff frequency

    process_stimuli(
        duration_ms=600,
        apply_filter=False,
        cutoff_hz=1000
    )