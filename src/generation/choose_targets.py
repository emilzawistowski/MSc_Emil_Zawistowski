import wave
import sys
from collections import defaultdict
from pathlib import Path
from datetime import datetime


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
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = OUTPUT_DIR / f"targets_report_{timestamp}.txt"


DATA_DIR = Path('/Users/emil/Desktop/MSc_Emil_Zawistowski/data/corpora/hillenbrand_et_al_1995-main/h95-alldata')
WAV_DIR = DATA_DIR / "men"                     
VOWDATA_FILE = DATA_DIR / "vowdata.dat"
TIMEDATA_FILE = DATA_DIR / "timedata.dat"
MISID_FILE = DATA_DIR / "misid.dat"

LP_CUTOFF = 800                                                          
TOP_SPEAKERS = 15
FOCUS_SPEAKERS = ["03", "07", "30"]
FOCUS_VOWELS = ["uw", "oo", "oa", "er", "ih", "iy", "eh", "uh"]

VOWEL_LABELS = {
    "ae": "had", "ah": "hod", "aw": "hawed", "eh": "head",
    "er": "heard", "ey": "hayed", "ih": "hid", "iy": "heed",
    "oa": "hoed", "oo": "hood", "uh": "hud", "uw": "who'd",
    "ei": "ei",
}


with open(log_file, "w", encoding="utf-8") as f_log:
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, f_log)


    def get_wav_duration_ms(filepath: Path) -> int | None:
        try:
            with wave.open(str(filepath), "r") as w:
                return round(w.getnframes() / w.getframerate() * 1000)
        except (wave.Error, OSError):
            return None

    wav_durations = {}
    for wav_path in WAV_DIR.rglob("*.wav"):
        dur = get_wav_duration_ms(wav_path)
        if dur is not None:
            wav_durations[wav_path.stem] = dur


    if not wav_durations:


        pass
    data = []
    with open(VOWDATA_FILE, "r", errors="ignore") as fh:
        for line in fh:
            parts = line.split()
            if not parts or parts[0][0] != "m":
                continue
            try:
                filename = parts[0]
                data.append({
                    "file": filename,
                    "talker": filename[1:3],
                    "vowel": filename[3:5],
                    "vow_dur": int(parts[1]), 
                    "f0": int(parts[2]),
                    "F1": int(parts[3]),
                    "F2": int(parts[4]),
                    "F3": int(parts[5]),
                    "file_dur": wav_durations.get(filename),
                })
            except (ValueError, IndexError):
                continue


    misid = set()
    with open(MISID_FILE, "r", errors="ignore") as fh:
        for line in fh:
            parts = line.split()
            if parts and parts[0][0] in "mwbg":
                misid.add(parts[0])


    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    def label(v):
        return VOWEL_LABELS.get(v, v)

    def fmt_dur(ms):
        return f"{ms:6}" if ms is not None else "   N/A"


    talker_f0 = defaultdict(list)
    talker_vowel = defaultdict(dict)
    for d in data:
        talker_f0[d["talker"]].append(d["f0"])
        talker_vowel[d["talker"]][d["vowel"]] = d

    ranked = sorted(
        ((t, mean(f0s), min(f0s)) for t, f0s in talker_f0.items()),
        key=lambda x: x[1],
    )


    for t, mf0, minf0 in ranked[:TOP_SPEAKERS]:
        vowels = ", ".join(sorted(talker_vowel[t]))


    vowel_agg = defaultdict(lambda: {"f0s": [], "F1s": [], "F2s": [], "vow_durs": [], "file_durs": []})
    for d in data:
        v = vowel_agg[d["vowel"]]
        v["f0s"].append(d["f0"])
        v["F1s"].append(d["F1"])
        v["vow_durs"].append(d["vow_dur"])
        if d["file_dur"] is not None:
            v["file_durs"].append(d["file_dur"])


    for v in sorted(vowel_agg):
        s = vowel_agg[v]
        fd = f"{mean(s['file_durs']):9.1f}" if s['file_durs'] else f"{'N/A':>9}"


    focus_tokens = [
        d for d in data
        if d["talker"] in FOCUS_SPEAKERS and d["vowel"] in FOCUS_VOWELS
    ]
    for d in sorted(focus_tokens, key=lambda x: x["f0"]):
        misid_flag = "[MISID]" if d["file"] in misid else ""
        status_flag = "FULL-BAND OK"


    for d in sorted((x for x in data if x["talker"] == "03"), key=lambda x: x["f0"]):
        note = "Natural timbre preserved (Full-band)"
        mflag = " [MISID]" if d["file"] in misid else ""


sys.stdout = original_stdout
