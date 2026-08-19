import wave
import sys
from collections import defaultdict
from pathlib import Path
from datetime import datetime

# ─── OUTPUT SETUP & LOGGING (Save to text file) ──────────────────────
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

# ─── CONFIG ──────────────────────────────────────────────────────────
DATA_DIR = Path('/Users/emil/Desktop/MSc_Emil_Zawistowski/data/corpora/hillenbrand_et_al_1995-main/h95-alldata')
WAV_DIR = DATA_DIR / "men" # .wav files for men
VOWDATA_FILE = DATA_DIR / "vowdata.dat"
TIMEDATA_FILE = DATA_DIR / "timedata.dat"
MISID_FILE = DATA_DIR / "misid.dat"

LP_CUTOFF = 800  # Left as a historical reference - filter is NOT applied
TOP_SPEAKERS = 15
FOCUS_SPEAKERS = ["03", "07", "30"]
FOCUS_VOWELS = ["uw", "oo", "oa", "er", "ih", "iy", "eh", "uh"]

VOWEL_LABELS = {
    "ae": "had", "ah": "hod", "aw": "hawed", "eh": "head",
    "er": "heard", "ey": "hayed", "ih": "hid", "iy": "heed",
    "oa": "hoed", "oo": "hood", "uh": "hud", "uw": "who'd",
    "ei": "ei",
}

# Redirect output (console + .txt file)
with open(log_file, "w", encoding="utf-8") as f_log:
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, f_log)

    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    print(f"Log will be saved to: {log_file}")
    print("-" * 78)

    # ─── WAV FILE DURATION ───────────────────────────────────────────────
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

    print(f"Found .wav files: {len(wav_durations)}")
    if not wav_durations:
        print("⚠ WARNING: no .wav files found – check WAV_DIR in config!")

    # ─── PARSE vowdata.dat (male speakers only) ──────────────────────────
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

    # ─── PARSE misid.dat ─────────────────────────────────────────────────
    misid = set()
    with open(MISID_FILE, "r", errors="ignore") as fh:
        for line in fh:
            parts = line.split()
            if parts and parts[0][0] in "mwbg":
                misid.add(parts[0])

    # ─── HELPERS ─────────────────────────────────────────────────────────
    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    def label(v):
        return VOWEL_LABELS.get(v, v)

    def fmt_dur(ms):
        return f"{ms:6}" if ms is not None else "   N/A"

    # ═══════════════════════════════════════════════════════════════════════════════
    # TABLE 1: Top speakers ranked by mean F0
    # ═══════════════════════════════════════════════════════════════════════════════
    talker_f0 = defaultdict(list)
    talker_vowel = defaultdict(dict)
    for d in data:
        talker_f0[d["talker"]].append(d["f0"])
        talker_vowel[d["talker"]][d["vowel"]] = d

    ranked = sorted(
        ((t, mean(f0s), min(f0s)) for t, f0s in talker_f0.items()),
        key=lambda x: x[1],
    )

    print("\n" + "=" * 78)
    print(f"TOP {TOP_SPEAKERS} MALE SPEAKERS BY LOWEST MEAN F0 (STABLE VOICE TARGETS)")
    print(f"{'Talker':8} {'Mean F0':>9} {'Min F0':>8} Vowels available")
    print("-" * 70)
    for t, mf0, minf0 in ranked[:TOP_SPEAKERS]:
        vowels = ", ".join(sorted(talker_vowel[t]))
        print(f"m{t:5} {mf0:9.1f} {minf0:8} {vowels}")

    # ═══════════════════════════════════════════════════════════════════════════════
    # TABLE 2: Vowel stats – F0 mean/min + vow dur + file dur
    # ═══════════════════════════════════════════════════════════════════════════════
    vowel_agg = defaultdict(lambda: {"f0s": [], "F1s": [], "F2s": [], "vow_durs": [], "file_durs": []})
    for d in data:
        v = vowel_agg[d["vowel"]]
        v["f0s"].append(d["f0"])
        v["F1s"].append(d["F1"])
        v["vow_durs"].append(d["vow_dur"])
        if d["file_dur"] is not None:
            v["file_durs"].append(d["file_dur"])

    print("\n" + "=" * 90)
    print("VOWEL ACOUSTIC STATS FOR MEN (Full-band analysis)")
    print(f"{'Vowel':6} {'Word':8} {'N':>4} {'F0 mean':>8} {'F0 min':>7} "
          f"{'Vow dur':>8} {'File dur':>9}")
    print("-" * 90)
    for v in sorted(vowel_agg):
        s = vowel_agg[v]
        fd = f"{mean(s['file_durs']):9.1f}" if s['file_durs'] else f"{'N/A':>9}"
        print(f"{v:6} {label(v):8} {len(s['f0s']):4} {mean(s['f0s']):8.1f} {min(s['f0s']):7} "
              f"{mean(s['vow_durs']):8.1f} {fd}")

    # ═══════════════════════════════════════════════════════════════════════════════
    # TABLE 3: Focus speakers × focus vowels
    # ═══════════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("FOCUS SPEAKERS x FOCUS VOWELS (Ranked by lowest F0 for maximum pitch stability)")
    print(f"{'File':10} {'Word':8} {'F0':>5} {'F1':>5} {'F2':>5} "
          f"{'Vow(ms)':>8} {'File(ms)':>9} {'Status':12} {'Misid'}")
    print("-" * 90)

    focus_tokens = [
        d for d in data
        if d["talker"] in FOCUS_SPEAKERS and d["vowel"] in FOCUS_VOWELS
    ]
    for d in sorted(focus_tokens, key=lambda x: x["f0"]):
        misid_flag = "[MISID]" if d["file"] in misid else ""
        status_flag = "FULL-BAND OK"
        print(f"{d['file']:10} {label(d['vowel']):8} {d['f0']:5} {d['F1']:5} "
              f"{d['F2']:5} {d['vow_dur']:8} {fmt_dur(d['file_dur']):>9} "
              f"{status_flag:12} {misid_flag}")

    # ═══════════════════════════════════════════════════════════════════════════════
    # TABLE 4: Final selection – m03, all full-band vowels
    # ═══════════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("FINAL SELECTION – speaker m03 (Full-band tokens, ranked by F0)")
    print(f"{'Token':10} {'Word':8} {'F0':>5} {'F1':>5} {'F2':>5} "
          f"{'Vow(ms)':>8} {'File(ms)':>9} {'Methodology Note':30}")
    print("-" * 90)
    for d in sorted((x for x in data if x["talker"] == "03"), key=lambda x: x["f0"]):
        note = "Natural timbre preserved (Full-band)"
        mflag = " [MISID]" if d["file"] in misid else ""
        print(f"{d['file']:10} {label(d['vowel']):8} {d['f0']:5} {d['F1']:5} "
              f"{d['F2']:5} {d['vow_dur']:8} {fmt_dur(d['file_dur']):>9} "
              f"{note:30}{mflag}")

# Restore standard output after exiting the 'with' block
sys.stdout = original_stdout
print(f"\n[Done] Report has been successfully saved to: {log_file}")