

import itertools
import numpy as np
import pandas as pd

from signalgen import params as default_params
from simulate_track import simulate_track
from person import run_tracker


def run_single_trial(
    target_range, target_velocity, drone_velocity, snr_db, clutter_power,
    n_frames=30, params=default_params, seed=None,
    n_required=3, m_window=5,
):
   
    det_df = simulate_track(
        initial_range=target_range,
        target_velocity=target_velocity,
        drone_velocity=drone_velocity,
        n_frames=n_frames,
        snr_db=snr_db,
        clutter_power=clutter_power,
        params=params,
        seed=seed,
        out_csv="_mc_detections_tmp.csv",
    )
    out_df = run_tracker(
        "_mc_detections_tmp.csv", "_mc_sim_output_tmp.csv",
        n_required=n_required, m_window=m_window,
    )

    error_rate = out_df["errorFlag"].mean()
    detection_rate = det_df["detected"].mean()
    ever_confirmed = bool(out_df["trackConfirmed"].any())

    return {
        "target_range": target_range,
        "target_velocity": target_velocity,
        "drone_velocity": drone_velocity,
        "snr_db": snr_db,
        "clutter_power": clutter_power,
        "error_rate": error_rate,
        "detection_rate": detection_rate,
        "track_confirmed": ever_confirmed,
    }


def monte_carlo_sweep(
    range_values=(60.0, 120.0, 180.0),
    velocity_values=(2.0, 8.0, 15.0),
    snr_values=(-45.0, -38.0, -30.0, -20.0),
    clutter_power=0.5,
    n_frames=30,
    n_trials_per_combo=5,
    params=default_params,
    n_required=3,
    m_window=5,
):
    """
    Full parameter sweep. For each (range, velocity, snr) combination,
    runs n_trials_per_combo independent trials (different random seeds)
    and averages the error rate.

    Returns a DataFrame, one row per parameter combination.
    """
    combos = list(itertools.product(range_values, velocity_values, snr_values))
    results = []

    print(f"Running Monte Carlo sweep: {len(combos)} combinations x {n_trials_per_combo} trials = "
          f"{len(combos) * n_trials_per_combo} total simulation runs...")

    for combo_i, (r, v, snr) in enumerate(combos):
        trial_error_rates = []
        trial_detection_rates = []
        for trial in range(n_trials_per_combo):
            seed = combo_i * 1000 + trial
            trial_result = run_single_trial(
                target_range=r,
                target_velocity=v,
                drone_velocity=0.0,
                snr_db=snr,
                clutter_power=clutter_power,
                n_frames=n_frames,
                params=params,
                seed=seed,
                n_required=n_required,
                m_window=m_window,
            )
            trial_error_rates.append(trial_result["error_rate"])
            trial_detection_rates.append(trial_result["detection_rate"])

        results.append({
            "target_range": r,
            "target_velocity": v,
            "snr_db": snr,
            "clutter_power": clutter_power,
            "mean_error_rate": np.mean(trial_error_rates) * 100,   # percent
            "std_error_rate": np.std(trial_error_rates) * 100,
            "mean_detection_rate": np.mean(trial_detection_rates) * 100,
            "n_trials": n_trials_per_combo,
        })

        print(f"  [{combo_i+1}/{len(combos)}] range={r}m v={v}m/s snr={snr}dB "
              f"-> error_rate={results[-1]['mean_error_rate']:.1f}%")

    return pd.DataFrame(results)


if __name__ == "__main__":
    summary = monte_carlo_sweep(
        range_values=(60.0, 120.0, 180.0),
        velocity_values=(2.0, 8.0, 15.0),
        snr_values=(-30.0, -25.0, -20.0, -15.0),
        clutter_power=0.5,
        n_frames=20,
        n_trials_per_combo=3,
    )

    summary.to_csv("monte_carlo_results.csv", index=False)

    overall_error_rate = summary["mean_error_rate"].mean()
    print(f"\nOverall mean error rate across all conditions: {overall_error_rate:.2f}%")
    print("Saved full sweep to monte_carlo_results.csv")

    print("\nWorst-performing conditions (highest error rate):")
    print(summary.sort_values("mean_error_rate", ascending=False).head(5).to_string(index=False))

    print("\nBest-performing conditions (lowest error rate):")
    print(summary.sort_values("mean_error_rate", ascending=True).head(5).to_string(index=False))
