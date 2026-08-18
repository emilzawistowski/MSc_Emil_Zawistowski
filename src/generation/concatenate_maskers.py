"""
Creates continuous masker streams.
"""

import json
import csv
import numpy as np
import soundfile as sf
from pathlib import Path
from datetime import datetime


DATASET_ROOT = Path(
    "/Users/emil/Desktop/MSc_Emil_Zawistowski/data/corpora/train-clean-100"
)
OUTPUT_DIR = Path(
    "/Users/emil/Desktop/MSc_Emil_Zawistowski/data/stimuli_selected/masker_stimuli"
)


SELECTED_SPEAKERS = ["3486", "4397", "2843", "78"]

TARGET_DURATION_S = 2400.0               
CROSSFADE_MS      = 30.0                                                    
MIN_SNR_DB        = 20.0                                                    
TARGET_RMS_DBFS   = -20.0                                           


def load_snr_map(speaker_dir: Path) -> dict[str, float]:
    """
    Reads SNR for each segment from .book.tsv files in the speaker folder.
    Returns {utterance_id: snr_db}.
    """
    snr_map = {}
    for tsv_path in speaker_dir.rglob("*.book.tsv"):
        with open(tsv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:

                if len(row) < 7 or row[0].startswith("#"):
                    continue
                try:
                    snr_map[row[0]] = float(row[6])
                except ValueError:
                    pass
    return snr_map


def crossfade_concat(chunks: list[np.ndarray], fade_samples: int) -> np.ndarray:
    """
    Concatenates a list of audio arrays using crossfade instead of adding silence.
    Every two adjacent segments overlap by fade_samples.

    Why better than silence+fade:
      - No distinct gaps (which can be acoustically salient themselves).
      - More resembles a natural, continuous speech stream.
      - Fade applies only to the interface — the interior of each segment is unchanged.
    """
    if not chunks:
        return np.array([], dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0]


    fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    fade_in  = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)

    result = chunks[0]
    for nxt in chunks[1:]:

        if len(result) < fade_samples or len(nxt) < fade_samples:
            result = np.concatenate([result, nxt])
            continue

        overlap = result[-fade_samples:] * fade_out + nxt[:fade_samples] * fade_in
        result  = np.concatenate([result[:-fade_samples], overlap, nxt[fade_samples:]])

    return result


def normalize_rms(audio: np.ndarray, target_dbfs: float) -> np.ndarray:
    """Normalizes RMS to a specified level in dBFS."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-9:
        return audio
    target_rms = 10 ** (target_dbfs / 20.0)
    return audio * (target_rms / rms)


def build_masker_stream(
    speaker_dir: Path,
    output_path: Path,
    target_duration_s: float = TARGET_DURATION_S,
    crossfade_ms: float      = CROSSFADE_MS,
    min_snr_db: float        = MIN_SNR_DB,
    target_rms_dbfs: float   = TARGET_RMS_DBFS,
) -> dict | None:
    """
    Builds one continuous masker stream for a single speaker.
    Returns a metadata dictionary or None in case of an error.
    """
    wav_files = sorted(speaker_dir.rglob("*.wav"))
    if not wav_files:

        return None


    snr_map = load_snr_map(speaker_dir)
    has_snr = bool(snr_map)

    chunks          = []
    used_files      = []
    skipped_snr     = 0
    total_loaded_s  = 0.0
    sample_rate     = None


    for wav_path in wav_files:

        utt_id = wav_path.stem
        if has_snr and utt_id in snr_map:
            if snr_map[utt_id] < min_snr_db:
                skipped_snr += 1
                continue

        data, sr = sf.read(wav_path, dtype="float32")


        if sample_rate is None:
            sample_rate = sr
            fade_samples = int(sr * crossfade_ms / 1000.0)
        elif sr != sample_rate:

            continue


        if data.ndim == 2:
            data = data.mean(axis=1)

        chunks.append(data)
        used_files.append(wav_path.name)
        total_loaded_s += len(data) / sr

        if total_loaded_s >= target_duration_s:
            break

    if not chunks:

        return None


    stream = crossfade_concat(chunks, fade_samples)


    stream = normalize_rms(stream, target_rms_dbfs)


    sf.write(output_path, stream, sample_rate)

    real_dur_s = len(stream) / sample_rate


    metadata = {
        "speaker_id":       speaker_dir.name,
        "output_file":      output_path.name,
        "sample_rate_hz":   sample_rate,
        "duration_s":       round(real_dur_s, 2),
        "segments_used":    len(chunks),
        "segments_skipped_snr": skipped_snr,
        "crossfade_ms":     crossfade_ms,
        "min_snr_db":       min_snr_db,
        "target_rms_dbfs":  target_rms_dbfs,
        "created_at":       datetime.now().isoformat(timespec="seconds"),
        "files_used":       used_files,
    }
    meta_path = output_path.with_suffix(".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


    return metadata


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


    speakers_tsv = DATASET_ROOT.parent / "speakers.tsv"
    gender_map: dict[str, str] = {}
    if speakers_tsv.exists():
        with open(speakers_tsv, newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) >= 2 and not row[0].startswith("#"):
                    gender_map[row[0].strip()] = row[1].strip()


    genders_in_selection = []
    for spk_id in SELECTED_SPEAKERS:
        g = gender_map.get(spk_id, "?")
        genders_in_selection.append(g)


    male_count   = genders_in_selection.count("M")
    female_count = genders_in_selection.count("F")
    if male_count != 2 or female_count != 2:


        pass
    all_meta = []
    for spk_id in SELECTED_SPEAKERS:
        speaker_dir = DATASET_ROOT / spk_id
        output_path = OUTPUT_DIR / f"masker_{spk_id}.wav"


        if not speaker_dir.exists():

            continue

        meta = build_masker_stream(
            speaker_dir      = speaker_dir,
            output_path      = output_path,
            target_duration_s= TARGET_DURATION_S,
            crossfade_ms     = CROSSFADE_MS,
            min_snr_db       = MIN_SNR_DB,
            target_rms_dbfs  = TARGET_RMS_DBFS,
        )
        if meta:
            all_meta.append(meta)


    summary_path = OUTPUT_DIR / "maskers_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, indent=2, ensure_ascii=False)
