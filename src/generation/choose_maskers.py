import parselmouth
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import sys
import json


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


OUTPUT_DIR = Path("experiment/stimuli_pipeline/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

log_file = OUTPUT_DIR / f"maskers_report_{timestamp}.txt"
json_file = OUTPUT_DIR / f"maskers_results_{timestamp}.json"


DATASET_ROOT = Path("/Users/emil/Desktop/MSc_Emil_Zawistowski/data/corpora/train-clean-100")
SPEAKERS_TSV = DATASET_ROOT / "speakers.tsv"


with open(log_file, "w", encoding="utf-8") as f:
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, f)


    male_speakers = set()
    with open(SPEAKERS_TSV) as fh:
        next(fh)
        for line in fh:
            parts = line.strip().split('\t')
            if len(parts) >= 3 and parts[1].upper() == "M" and parts[2] == "train-clean-100":
                male_speakers.add(parts[0])


    def measure_f0(wav_path):
        try:
            snd = parselmouth.Sound(str(wav_path))
            pitch = snd.to_pitch(pitch_floor=60, pitch_ceiling=200)
            values = pitch.selected_array["frequency"]
            values = values[values > 0]
            if len(values) == 0:
                return None
            return float(np.median(values))
        except Exception as e:

            return None

    def get_real_duration_min(wav_files):
        total = 0.0
        for wav in wav_files:
            try:
                total += sf.info(wav).duration
            except Exception as e:

                pass
        return total / 60.0


    results = {}

    for spk_id in tqdm(sorted(male_speakers), desc="Analyzing speakers"):
        spk_dir = DATASET_ROOT / spk_id
        if not spk_dir.exists():
            continue

        wav_files = sorted(spk_dir.rglob("*.wav"))
        if len(wav_files) < 3:
            continue

        f0_values = []
        for wav in wav_files:
            f0 = measure_f0(wav)
            if f0 is not None:
                f0_values.append(f0)

        if not f0_values:
            continue

        real_duration_min = get_real_duration_min(wav_files)

        results[spk_id] = {
            "mean_f0": float(np.mean(f0_values)),
            "f0_std": float(np.std(f0_values)),
            "duration_min": real_duration_min,
            "n_files": len(wav_files)
        }


    MIN_DURATION_MIN = 8.0


    filtered = [(sid, d) for sid, d in results.items() if d["duration_min"] >= MIN_DURATION_MIN]
    ranked = sorted(filtered, key=lambda x: x[1]["mean_f0"])


    for rank, (sid, d) in enumerate(ranked[:25], 1):


        pass
    for sid in ["3486", "4397", "2843", "78"]:
        if sid in results:
            d = results[sid]


        else:


            pass
    with open(json_file, "w", encoding="utf-8") as jf:
        json.dump(results, jf, indent=2, ensure_ascii=False)


    sys.stdout = original_stdout