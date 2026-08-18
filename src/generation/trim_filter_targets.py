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


            n_samples = int((duration_ms / 1000.0) * sr)


            trimmed_data = data[:n_samples]


            if len(trimmed_data) < n_samples:
                pad_length = n_samples - len(trimmed_data)


                if trimmed_data.ndim == 1:
                    trimmed_data = np.pad(
                        trimmed_data,
                        (0, pad_length),
                        mode='constant'
                    )


                else:
                    trimmed_data = np.pad(
                        trimmed_data,
                        ((0, pad_length), (0, 0)),
                        mode='constant'
                    )


            if apply_filter:
                nyquist = 0.5 * sr
                normal_cutoff = cutoff_hz / nyquist


                b, a = butter(
                    N=2,
                    Wn=normal_cutoff,
                    btype='low',
                    analog=False
                )


                trimmed_data = lfilter(b, a, trimmed_data, axis=0)


            fade_len = int(0.005 * sr)

            if len(trimmed_data) > fade_len:
                fade_curve = np.linspace(1.0, 0.0, fade_len)


                if trimmed_data.ndim == 1:
                    trimmed_data[-fade_len:] *= fade_curve


                else:
                    trimmed_data[-fade_len:] *= fade_curve[:, np.newaxis]


            if len(trimmed_data) > 2 * fade_len:

                fade_in_curve = np.linspace(0.0, 1.0, 2 * fade_len)


                if trimmed_data.ndim == 1:
                    trimmed_data[:fade_len * 2] *= fade_in_curve

                else:
                    trimmed_data[:fade_len * 2] *= fade_in_curve[:, np.newaxis]


            sf.write(out_path, trimmed_data, sr)

            filter_status = (
                f"LP {cutoff_hz}Hz"
                if apply_filter
                else "No filter"
            )


        except Exception as e:


            pass
if __name__ == "__main__":


    process_stimuli(
        duration_ms=600,
        apply_filter=False,
        cutoff_hz=1000
    )