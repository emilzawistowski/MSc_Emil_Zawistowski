# Auditory Distance Perception in Wave Field Synthesis: Pre-Attentive Neural Processing in a Cocktail-Party Environment

Code and data for my MSc thesis at Aalborg University.

The thesis looks at how the human auditory system encodes changes in source distance in a multi-talker environment. It combines Wave Field Synthesis (60-loudspeaker array) with 32-channel EEG, and tests a few hypotheses derived from predictive coding: automatic deviance detection (MMN), looming asymmetry, involuntary attention capture (P3a), and single-trial decoding of distance representations.

## Repository layout

```
src/         project code: EEG analysis, stimulus generation, measurement scripts, utilities
notebooks/   demo notebooks
data/        measurements and metadata (large datasets are git-ignored)
results/     figures, tables, reports
```

## Setup

Conda (recommended):

```bash
conda env create -f environment.yml
conda activate thesis_eeg
```

or pip:

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

Should work on Windows, Linux and macOS - paths go through `pathlib` and plots use the Agg backend, so the scripts also run headless.

## Analysis pipeline

First, set everything up with `src/config.py`.

Then run in order:

```bash
python src/analysis/00a_raw_noise_check.py       # raw recording quality screening
python src/analysis/00b_flag_bad_recordings.py   # flag and exclude noisy/bad recordings
python src/analysis/01_preprocessing.py          # filtering, ICA, epoching -> .fif
python src/analysis/02_erp_analysis.py           # grand averages, MMN/P3a difference waves
python src/analysis/03_statistics.py             # cluster permutation tests, Bayes factors
python src/analysis/04_decoding.py               # single-trial SVM decoding
python src/analysis/04b_decoding_csp_lda.py      # CSP+LDA, leave-one-subject-out
```

There are also helper scripts in `src/analysis/` for exploratory diagnostics, including `05_signal_diagnostics.py` and `06_power_analysis.py` (post-hoc sensitivity/MDE analysis), as well as `manual_eeg_browser.py` for browsing raw signals and epochs by hand.

## Data

Raw EEG, processed epochs and the speech corpora are too big for GitHub and are listed in `.gitignore`. The repo ships the stimuli metadata, trigger logs and acoustic measurements needed to reproduce the analysis outputs. If you need the full dataset, get in touch.

---

Emil Zawistowski
