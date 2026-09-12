

import numpy as np
import pandas as pd

from signalgen import params, generate_clean_signal
from final import detectRange


def run_frame(target_range, target_velocity, drone_velocity, params, snr_db, clutter_power):
    """Run one frame through the full A -> B pipeline and return a detection dict."""
    beat_signal = generate_clean_signal(target_range, target_velocity, drone_velocity, params)
    

    detection = detectRange(beat_signal, params, snr_db=snr_db, clutter_power=clutter_power)
    
    return detection["detected"], detection["measuredRange"], detection["snr_dB"]

def simulate_track(
    initial_range=120.0,
    target_velocity=8.0,
    drone_velocity=0.0,
    n_frames=40,
    snr_db=10,
    clutter_power=0.1,
    params=params,
    seed=None,
    out_csv="detections.csv",
):
    
    if seed is not None:
        np.random.seed(seed)

    dt = params["N"] * params["Tc"]  # one frame = one full chirp burst (CPI)
    relative_velocity = target_velocity - drone_velocity

    rows = []
    for i in range(n_frames):
        frame_time = i * dt
        true_range = initial_range + relative_velocity * frame_time

        detected, measured_range, snr_est = run_frame(
            true_range, target_velocity, drone_velocity, params, snr_db, clutter_power
        )

        rows.append({
            "frameTime": frame_time,
            "trueRange": true_range,
            "detected": int(detected),
            "measuredRange": measured_range if detected else np.nan,
            "snr_dB": snr_est if detected else np.nan,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


if __name__ == "__main__":
    print("Running multi-frame")
    df = simulate_track(
        initial_range=120.0,
        target_velocity=8.0,
        drone_velocity=0.0,
        n_frames=40,
        snr_db=10,
        clutter_power=0.1,
        seed=0,
    )
    print(df.head(10))
    print(f"\n{df['detected'].sum()} / {len(df)} frames detected")
    print("Saved to detections.csv")
