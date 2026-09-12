

import numpy as np
import pandas as pd


# ============================================================
# 1. KALMAN FILTER (2-state: range, range_rate)
# ============================================================

def kalman_predict(x_est, P_est, dt, Q):
    """
    Predict step for constant-velocity model.

    State: x = [range; range_rate]
    Transition: range_k+1 = range_k + range_rate_k * dt
                range_rate_k+1 = range_rate_k
    """
    F = np.array([[1.0, dt],
                  [0.0, 1.0]])

    x_pred = F @ x_est
    P_pred = F @ P_est @ F.T + Q
    return x_pred, P_pred


def kalman_update(x_est, P_est, measurement, detected, dt, Q, R):
    """
    Full predict + (conditional) correct step.

    Inputs
    ------
    x_est, P_est : previous state estimate [range, range_rate] and covariance (2x2)
    measurement  : measuredRange from detectRange() (ignored if detected == False)
    detected     : bool, whether to run the correction step this frame
    dt           : frame interval, seconds
    Q            : process noise covariance (2x2)
    R            : measurement noise variance (scalar, range measurement only)

    Output
    ------
    x_new, P_new : updated (or predict-only) state estimate and covariance
    """
    # --- Predict ---
    x_pred, P_pred = kalman_predict(x_est, P_est, dt, Q)

    if not detected:
        # No measurement this frame -- coast on the prediction only.
        return x_pred, P_pred

    # --- Correct (measurement update) ---
    H = np.array([[1.0, 0.0]])  # we only measure range, not range_rate
    z = np.array([measurement])

    y = z - H @ x_pred                      # innovation
    S = H @ P_pred @ H.T + R                 # innovation covariance
    K = P_pred @ H.T @ np.linalg.inv(S)      # Kalman gain (2x1)

    x_new = x_pred + (K.flatten() * y[0])
    P_new = (np.eye(2) - K @ H) @ P_pred

    return x_new, P_new


def default_Q(dt, process_noise_std=0.5):
    """
    Simple constant-velocity process noise model (discretized white-noise
    acceleration). process_noise_std is in m/s^2.
    """
    q = process_noise_std ** 2
    Q = q * np.array([
        [dt**4 / 4, dt**3 / 2],
        [dt**3 / 2, dt**2],
    ])
    return Q


# ============================================================
# 2. TRACK CONFIRMATION (N-of-M logic)
# ============================================================

class TrackConfirmationLogic:
    """
    Require N detections out of the last M frames before a track counts
    as 'confirmed'. Once confirmed, the track stays confirmed unless it
    drops out of detection for too long (M consecutive misses), at which
    point it must re-confirm.
    """

    def __init__(self, n_required=3, m_window=5, drop_after_misses=None):
        if n_required > m_window:
            raise ValueError(
                f"n_required ({n_required}) cannot exceed m_window ({m_window}) -- "
                f"a track can never get {n_required} hits inside a {m_window}-frame window. "
                f"Set N <= M."
            )
        self.n_required = n_required
        self.m_window = m_window
        self.drop_after_misses = drop_after_misses or m_window
        self.history = []       # rolling window of detected flags (bool)
        self.confirmed = False
        self.consecutive_misses = 0

    def update(self, detected):
        self.history.append(bool(detected))
        if len(self.history) > self.m_window:
            self.history.pop(0)

        if detected:
            self.consecutive_misses = 0
        else:
            self.consecutive_misses += 1

        hits_in_window = sum(self.history)

        if not self.confirmed:
            if hits_in_window >= self.n_required:
                self.confirmed = True
        else:
            if self.consecutive_misses >= self.drop_after_misses:
                self.confirmed = False
                self.history = []

        return self.confirmed


# ============================================================
# 3. RUN TRACKER OVER A FULL detections.csv
# ============================================================

def run_tracker(
    detections_csv="detections.csv",
    out_csv="sim_output.csv",
    n_required=3,
    m_window=5,
    process_noise_std=15.0,
    measurement_noise_std=0.3,
    error_range_threshold=5.0,
):
    """
    Reads detections.csv (Section 4.2 format), runs the Kalman filter +
    N-of-M track confirmation across every frame, and writes
    sim_output.csv (Section 4.3 format).

    NOTE on default Q/R: one "frame" here is one full CPI (N chirps), which
    with this waveform's params (N=128, Tc=10us) is only ~1.28ms long. At a
    typical drone-relevant closing speed (5-20 m/s), the target moves only
    ~1-3cm between frames -- far below a naive 1m measurement-noise
    assumption. If R is left too loose relative to that per-frame motion,
    the filter cannot see velocity at all and the Kalman range stays flat
    while the true range visibly moves (this was diagnosed from a real
    dashboard run). measurement_noise_std is instead set close to the
    CFAR range-bin precision (fractions of a meter), and process_noise_std
    is raised so the filter trusts new measurements enough to pick up
    velocity within a few dozen frames.

    errorFlag definition (per project plan Section 1):
        1 if this frame counts as a detection error, i.e.:
          - a missed detection (target present, CFAR did not detect), or
          - a false detection / bad measurement (CFAR detected, but the
            measured range is off from ground truth by more than
            error_range_threshold meters)
    """
    df = pd.read_csv(detections_csv)

    frame_times = df["frameTime"].to_numpy()
    dts = np.diff(frame_times, prepend=frame_times[0])
    dts[0] = dts[1] if len(dts) > 1 else 0.001  # avoid dt=0 on first frame

    # --- Initialize Kalman state from the first detection ---
    first_detected_idx = df.index[df["detected"] == 1]
    if len(first_detected_idx) == 0:
        # No detections anywhere in this run (can happen at very low SNR
        # during a Monte Carlo sweep). Every frame is a miss / error.
        rows = []
        for _, row in df.iterrows():
            rows.append({
                "frameTime": row["frameTime"],
                "trueRange": row["trueRange"],
                "measuredRange": np.nan,
                "kalmanRange": np.nan,
                "trackConfirmed": 0,
                "errorFlag": 1,
            })
        out_df = pd.DataFrame(rows)
        out_df.to_csv(out_csv, index=False)
        return out_df

 # --- Delay Kalman state initialization until first actual detection ---
    x_est = None  
    P_est = None
    
    R = np.array([[measurement_noise_std ** 2]])
    tracker = TrackConfirmationLogic(n_required=n_required, m_window=m_window)

    rows = []
    for i, row in df.iterrows():
        dt = dts[i] if dts[i] > 0 else 1e-6
        Q = default_Q(dt, process_noise_std)

        detected = bool(row["detected"])
        measurement = row["measuredRange"] if detected else np.nan
        true_range = row["trueRange"]

        # Dynamic initialization logic
        if x_est is None:
            if detected:
                # Initialize state dynamically on the exact frame it's first detected
                x_est = np.array([measurement, 0.0]) 
                P_est = np.diag([25.0, 25.0])
                kalman_range = x_est[0]
            else:
                kalman_range = np.nan
        else:
            # Normal update if already initialized
            x_est, P_est = kalman_update(x_est, P_est, measurement, detected, dt, Q, R)
            kalman_range = x_est[0]

        confirmed = tracker.update(detected)

        # Error flag: missed detection, OR false/bad measurement
        if not detected:
            error_flag = 1
        else:
            error_flag = int(abs(measurement - true_range) > error_range_threshold)

        rows.append({
            "frameTime": row["frameTime"],
            "trueRange": true_range,
            "measuredRange": measurement,
            "kalmanRange": kalman_range,
            "trackConfirmed": int(confirmed),
            "errorFlag": error_flag,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)
    return out_df


if __name__ == "__main__":
    print("Running Kalman tracker + track confirmation over detections.csv...")
    out_df = run_tracker("detections.csv", "sim_output.csv")
    print(out_df.head(10))
    error_rate = out_df["errorFlag"].mean() * 100
    print(f"\nError rate: {error_rate:.2f}%")
    print(f"Frames with confirmed track: {out_df['trackConfirmed'].sum()} / {len(out_df)}")
    print("Saved to sim_output.csv")
