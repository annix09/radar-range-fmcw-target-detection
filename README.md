# FMCW Drone Radar — Moving Target Range Detection

A simulation pipeline and interactive dashboard that detects the range from a moving target to a drone using FMCW (Frequency-Modulated Continuous Wave) radar, targeting **<5% detection error** across realistic noise, clutter, and velocity conditions.

The pipeline simulates the full radar chain — chirp generation, propagation, receiver noise, ground clutter, CFAR detection, and Kalman-filtered tracking — then validates performance with a Monte Carlo sweep and visualizes everything in a live Streamlit dashboard.

**Live dashboard:** _add your Streamlit Cloud URL here once deployed_

---

## Pipeline overview

```
signalgen.py  →  final.py  →  simulate_track.py  →  person.py  →  dashboard.py
  (chirp +        (noise,        (multi-frame          (Kalman          (Streamlit
   beat signal)    clutter,       loop → CSV)            tracking +       UI)
                   CFAR)                                 N-of-M logic)
```

| Stage | File | Role |
|---|---|---|
| 1 | `signalgen.py` | Generates the clean FMCW chirp train and beat signal for one target, given range, velocity, and drone velocity. Includes range/Doppler FFT validation against ground truth. |
| 2 | `final.py` | Adds realistic receiver noise and ground clutter to the clean signal, runs range-Doppler FFT processing + MTI clutter filtering, and runs CA-CFAR detection to estimate range and SNR per frame. |
| 3 | `simulate_track.py` | Runs the signal → detection pipeline frame-by-frame over a moving target trajectory, exporting raw per-frame detections to `detections.csv`. |
| 4 | `person.py` | Reads `detections.csv`, applies a 2-state (range, range-rate) Kalman filter plus N-of-M track confirmation logic, flags detection errors, and exports `sim_output.csv`. |
| 5 | `monte_carlo.py` | Sweeps target range, velocity, SNR, and clutter power across many trials to compute mean error rate per condition, validating the <5% target. |
| 6 | `dashboard.py` | Streamlit app — run simulations interactively, view range-time plots, error-rate gauge, and a range-Doppler heatmap, or upload an existing `sim_output.csv`. |

---

## Team

| Person | Role | Owns |
|---|---|---|
| **Person A** | Signal & Waveform | `signalgen.py` — parameter design, chirp generation, clean beat signal, range FFT validation |
| **Sumir — Person B** | Propagation, Noise, Clutter & Detection | `final.py` — receiver noise, ground clutter, MTI filtering, CFAR detection |
| **Udai — Person C** | Tracking, Validation & Dashboard | `person.py`, `monte_carlo.py`, `dashboard.py` — Kalman tracking, track confirmation, Monte Carlo validation, dashboard |

---

## Getting started

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/radar-range-fmcw-target-detection.git
cd radar-range-fmcw-target-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the pipeline from the command line (optional — sanity check)

```bash
python signalgen.py        # validates the clean signal + range FFT, saves a reference plot
python simulate_track.py   # runs a multi-frame simulation, saves detections.csv
python person.py           # runs the Kalman tracker, saves sim_output.csv
python monte_carlo.py      # runs the full validation sweep, saves monte_carlo_results.csv
```

### 4. Launch the dashboard locally

```bash
streamlit run dashboard.py
```

This opens the dashboard at `http://localhost:8501`. Use the sidebar to set target range, velocity, SNR, and clutter power, then click **Run simulation + tracker** — or upload an existing `sim_output.csv`.

---

## Data format reference

### `detections.csv` (output of `simulate_track.py`, input to `person.py`)

| Column | Type | Notes |
|---|---|---|
| `frameTime` | float | seconds, simulation time |
| `trueRange` | float | meters, ground-truth target range |
| `detected` | 0/1 | 1 if CFAR flagged a detection this frame |
| `measuredRange` | float | meters, raw CFAR-detected range (blank if not detected) |
| `snr_dB` | float | estimated SNR at detected peak (blank if not detected) |

### `sim_output.csv` (output of `person.py`, input to `dashboard.py`)

| Column | Type | Notes |
|---|---|---|
| `frameTime` | float | seconds |
| `trueRange` | float | meters, ground truth |
| `measuredRange` | float | meters, raw CFAR detection (blank if missed) |
| `kalmanRange` | float | meters, filtered/tracked range estimate |
| `trackConfirmed` | 0/1 | 1 if track passed N-of-M confirmation logic |
| `errorFlag` | 0/1 | 1 if this frame counts as a detection error (missed or false) |

**Error rate** = mean of `errorFlag` across all frames — this is the metric being validated against the <5% target.

---

## Deployment

The dashboard is deployed on [Streamlit Community Cloud](https://streamlit.io/cloud), which auto-redeploys on every push to `main`.

- Entry point: `dashboard.py`
- Dependencies: `requirements.txt`
- If you fork/redeploy this yourself: point Streamlit Cloud at your repo, branch `main`, main file `dashboard.py`.

---

## Repo structure

```
.
├── signalgen.py        # Person A — waveform + clean signal
├── final.py             # Sumir — noise, clutter, CFAR detection
├── simulate_track.py    # multi-frame simulation loop
├── person.py             # Udai — Kalman tracker + track confirmation
├── monte_carlo.py        # Udai — validation sweep
├── dashboard.py           # Udai — Streamlit dashboard
├── requirements.txt
├── runtime.txt            # pins Python version for deployment (if present)
└── README.md
```
