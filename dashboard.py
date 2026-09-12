

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from signalgen import params as default_params, generate_clean_signal
from final import add_receiver_noise, add_ground_clutter, range_doppler_processing, apply_mti
from simulate_track import simulate_track
from person import run_tracker

st.set_page_config(page_title="FMCW Drone Radar Dashboard", layout="wide")

st.title("FMCW Drone Radar — Range Detection Dashboard")
st.caption("Kalman tracking, N-of-M confirmation, and Monte Carlo error-rate validation")

# ---------------------------------------------------------------------------
# Sidebar controls — generate a fresh run, or load an existing CSV
# ---------------------------------------------------------------------------
st.sidebar.header("Run a New Simulation")

target_range = st.sidebar.slider("Initial target range (m)", 20.0, 220.0, 120.0, step=5.0)
target_velocity = st.sidebar.slider("Target velocity (m/s)", -30.0, 30.0, 8.0, step=0.5)
drone_velocity = st.sidebar.slider("Drone velocity (m/s)", -30.0, 30.0, 0.0, step=0.5)
n_frames = st.sidebar.slider("Number of frames", 10, 100, 40, step=5)
snr_db = st.sidebar.slider("Receiver SNR (dB)", -50, 30, -30, step=1,
                            help="This waveform has very high coherent processing gain (N=128 chirps), "
                                 "so realistic SNR sweep range is roughly -45 to -20 dB.")
clutter_power = st.sidebar.slider("Ground clutter power", 0.0, 1.0, 0.5, step=0.05)

st.sidebar.header("Tracker Settings")
m_window = st.sidebar.number_input("M (window size)", value=5, min_value=2, max_value=20, step=1)
n_required = st.sidebar.slider(
    "N (hits required out of M)", min_value=1, max_value=int(m_window), value=min(3, int(m_window)), step=1,
    help="Must be <= M. A common choice is roughly 60-70% of M."
)

run = st.sidebar.button("Run simulation + tracker", type="primary", use_container_width=True)

st.sidebar.divider()
st.sidebar.header("Or Load an Existing sim_output.csv")
uploaded = st.sidebar.file_uploader("Upload sim_output.csv", type="csv")

# ---------------------------------------------------------------------------
# Run / load
# ---------------------------------------------------------------------------
if "sim_df" not in st.session_state:
    st.session_state.sim_df = None

if run:
    with st.spinner("Simulating frames and running Kalman tracker..."):
        det_df = simulate_track(
            initial_range=target_range,
            target_velocity=target_velocity,
            drone_velocity=drone_velocity,
            n_frames=n_frames,
            snr_db=snr_db,
            clutter_power=clutter_power,
            out_csv="detections.csv",
        )
        out_df = run_tracker("detections.csv", "sim_output.csv", n_required=n_required, m_window=m_window)
        st.session_state.sim_df = out_df
        st.session_state.last_params = dict(
            target_range=target_range, target_velocity=target_velocity,
            drone_velocity=drone_velocity, snr_db=snr_db, clutter_power=clutter_power,
        )

if uploaded is not None:
    st.session_state.sim_df = pd.read_csv(uploaded)

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
if st.session_state.sim_df is None:
    st.info("Click **Run simulation + tracker** in the sidebar, or upload an existing `sim_output.csv`.")
else:
    df = st.session_state.sim_df

    # --- Error rate gauge (as metrics, since a true gauge needs plotly) ---
    st.subheader("Error Rate Summary")
    error_rate = df["errorFlag"].mean() * 100
    detection_rate = df["measuredRange"].notna().mean() * 100
    confirm_rate = df["trackConfirmed"].mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Error rate", f"{error_rate:.1f}%", delta=f"{error_rate - 5:.1f} pts vs 5% target", delta_color="inverse")
    c2.metric("Detection rate", f"{detection_rate:.1f}%")
    c3.metric("Track confirmed", f"{confirm_rate:.1f}% of frames")
    c4.metric("Total frames", f"{len(df)}")

    # Gauge-style bar
    fig_gauge, ax_gauge = plt.subplots(figsize=(9, 0.9))
    gauge_color = "#2C5F2D" if error_rate <= 5 else ("#E8A33D" if error_rate <= 25 else "#B7472A")
    ax_gauge.barh([0], [error_rate], color=gauge_color, height=0.6)
    ax_gauge.barh([0], [100], color="#EAEAEA", height=0.6, zorder=0)
    ax_gauge.barh([0], [error_rate], color=gauge_color, height=0.6, zorder=1)
    ax_gauge.axvline(5, color="black", linestyle="--", linewidth=1, label="5% target")
    ax_gauge.set_xlim(0, 100)
    ax_gauge.set_yticks([])
    ax_gauge.set_xlabel("Error rate (%)")
    ax_gauge.legend(loc="upper right", fontsize=8)
    st.pyplot(fig_gauge)
    plt.close(fig_gauge)

    st.divider()

    # --- Range-time plot ---
    st.subheader("Range-Time Plot")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(df["frameTime"], df["trueRange"], label="True range", color="#2C5F2D", linewidth=2)
    ax1.scatter(df["frameTime"], df["measuredRange"], label="Raw CFAR measurement",
                color="#B7472A", s=18, alpha=0.7, zorder=3)
    ax1.plot(df["frameTime"], df["kalmanRange"], label="Kalman-filtered range",
              color="#065A82", linewidth=2, linestyle="--")

    confirmed_mask = df["trackConfirmed"] == 1
    if confirmed_mask.any():
        ax1.fill_between(df["frameTime"], ax1.get_ylim()[0], ax1.get_ylim()[1],
                          where=confirmed_mask, color="#065A82", alpha=0.06,
                          label="Track confirmed", step="mid")

    ax1.set_xlabel("Frame time (s)")
    ax1.set_ylabel("Range (m)")
    ax1.set_title("True vs Measured vs Kalman-Filtered Range")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(alpha=0.3)
    st.pyplot(fig1)
    plt.close(fig1)

    # --- Error flag timeline ---
    st.subheader("Error Flag Timeline")
    fig2, ax2 = plt.subplots(figsize=(10, 1.6))
    colors = ["#B7472A" if e else "#2C5F2D" for e in df["errorFlag"]]
    ax2.bar(df["frameTime"], [1] * len(df), color=colors, width=(df["frameTime"].iloc[1] - df["frameTime"].iloc[0]) * 0.9
            if len(df) > 1 else 0.001)
    ax2.set_yticks([])
    ax2.set_xlabel("Frame time (s)")
    ax2.set_title("Green = correct frame · Red = error (missed or false detection)")
    st.pyplot(fig2)
    plt.close(fig2)

    st.divider()

    # --- Range-Doppler heatmap (fresh single-frame context view) ---
    st.subheader("Range-Doppler Map (Reference Frame)")
    st.caption("Single-frame snapshot using the same target/velocity settings, for visual context.")
    p = st.session_state.get("last_params", dict(
        target_range=target_range, target_velocity=target_velocity,
        drone_velocity=drone_velocity, snr_db=snr_db, clutter_power=clutter_power,
    ))
    beat_signal = generate_clean_signal(p["target_range"], p["target_velocity"], p["drone_velocity"], default_params)
    noisy_signal = add_receiver_noise(beat_signal, p["snr_db"])
    cluttered_signal = add_ground_clutter(noisy_signal, p["clutter_power"])
    ranges, velocities, range_doppler = range_doppler_processing(cluttered_signal, default_params)
    filtered_rd = apply_mti(range_doppler, velocities)

    fig3, ax3 = plt.subplots(figsize=(9, 4))
    extent = [velocities.min(), velocities.max(), ranges.min(), ranges.max()]
    im = ax3.imshow(np.abs(filtered_rd), aspect="auto", origin="lower", extent=extent, cmap="viridis")
    ax3.set_xlabel("Relative velocity (m/s)")
    ax3.set_ylabel("Range (m)")
    ax3.set_title("Range-Doppler Map (after MTI)")
    fig3.colorbar(im, ax=ax3, label="Magnitude")
    st.pyplot(fig3)
    plt.close(fig3)

    st.divider()

    # --- Raw data table ---
    with st.expander("Show raw sim_output.csv data"):
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download sim_output.csv",
            data=df.to_csv(index=False),
            file_name="sim_output.csv",
            mime="text/csv",
        )
