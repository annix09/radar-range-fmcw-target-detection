"""
FMCW radar signal generation for drone radar project.
Person A deliverable (Section 4.1 interface):

    beatSignal = generateCleanSignal(targetRange, targetVelocity, droneVelocity, params)

Output: [Nsamples x N] matrix, clean (noise-free) beat signal, N = chirps/frame.
Hands off to Person B's detectRange(beatSignal, params).
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# params struct/dict — matches the doc's field names exactly
# ---------------------------------------------------------------------------
params = {
    "fc": 24e9,     # center frequency, Hz
    "B": 150e6,     # sweep bandwidth, Hz
    "Tc": 10e-6,    # chirp duration, s
    "Fs": 50e6,     # sample rate, Hz
    "N": 128,       # chirps per frame
}

c = 3e8  # speed of light, m/s


def design_metrics(params):
    """Range resolution and max unambiguous range from the waveform design."""
    dR = c / (2 * params["B"])
    Rmax = (params["Fs"] * c * params["Tc"]) / (4 * params["B"])
    return dR, Rmax


def generate_clean_signal(target_range, target_velocity, drone_velocity, params):
    """
    Simulate a clean (noise-free) FMCW chirp train for one point target.

    Parameters
    ----------
    target_range : float
        Initial target range, meters.
    target_velocity : float
        Target radial velocity, m/s (+ away from drone).
    drone_velocity : float
        Drone's own radial velocity, m/s (+ away from target).
    params : dict
        fc, B, Tc, Fs, N

    Returns
    -------
    beat_signal : np.ndarray, complex, shape [Nsamples x N]
        Column i = beat signal for chirp i.
    """
    fc, B, Tc, Fs, N = params["fc"], params["B"], params["Tc"], params["Fs"], params["N"]

    n_samples = int(Fs * Tc)
    t = np.linspace(0, Tc, n_samples, endpoint=False)
    k = B / Tc  # chirp rate, Hz/s

    # Relative closing velocity (positive = target moving away from drone)
    relative_velocity = target_velocity - drone_velocity

    beat_signal = np.zeros((n_samples, N), dtype=complex)

    for i in range(N):
        # Range at the start of chirp i, assuming constant relative velocity
        chirp_start_time = i * Tc
        range_i = target_range + relative_velocity * chirp_start_time

        tau = 2 * range_i / c  # round-trip delay for this chirp

        tx = np.exp(1j * 2 * np.pi * (fc * t + 0.5 * k * t**2))
        rx = np.exp(1j * 2 * np.pi * (fc * (t - tau) + 0.5 * k * (t - tau) ** 2))

        beat_signal[:, i] = tx * np.conj(rx)

    return beat_signal


def range_doppler_validate(beat_signal, params, ground_truth_range):
    """
    2D FFT: range FFT per chirp (fast time), then Doppler FFT across chirps
    (slow time). Returns estimated range/velocity plus arrays for plotting.
    """
    fc, B, Tc, Fs, N = params["fc"], params["B"], params["Tc"], params["Fs"], params["N"]
    n_samples = beat_signal.shape[0]

    # --- Range FFT (fast time, per chirp) ---
    window_range = np.hanning(n_samples)[:, None]
    range_fft = np.fft.fft(beat_signal * window_range, axis=0)

    half = n_samples // 2
    range_fft = range_fft[:half, :]
    freqs_range = np.fft.fftfreq(n_samples, d=1 / Fs)[:half]
    ranges = (freqs_range * c * Tc) / (2 * B)

    # --- Doppler FFT (slow time, across chirps) ---
    window_doppler = np.hanning(N)[None, :]
    range_doppler = np.fft.fftshift(np.fft.fft(range_fft * window_doppler, axis=1), axes=1)
    freqs_doppler = np.fft.fftshift(np.fft.fftfreq(N, d=Tc))
    wavelength = c / fc
    velocities = freqs_doppler * wavelength / 2

    mag = np.abs(range_doppler)
    peak_flat = np.argmax(mag)
    range_idx, doppler_idx = np.unravel_index(peak_flat, mag.shape)

    est_range = ranges[range_idx]
    est_velocity = velocities[doppler_idx]

    print(f"Ground-truth range:     {ground_truth_range:.2f} m")
    print(f"Estimated range:        {est_range:.2f} m")
    print(f"Estimated rel. velocity:{est_velocity:.2f} m/s")

    return ranges, velocities, mag, est_range, est_velocity


if __name__ == "__main__":
    dR, Rmax = design_metrics(params)
    print(f"Range resolution (dR): {dR:.2f} m")
    print(f"Max unambiguous range (R_max): {Rmax:.2f} m\n")

    # --- Test case: target at 120 m, moving away at 8 m/s, drone stationary ---
    target_range = 120.0
    target_velocity = 8.0
    drone_velocity = 0.0

    beat_signal = generate_clean_signal(target_range, target_velocity, drone_velocity, params)
    print(f"beat_signal shape: {beat_signal.shape}  (expect [Nsamples x N] = "
          f"[{int(params['Fs']*params['Tc'])} x {params['N']}])\n")

    ranges, velocities, mag, est_range, est_velocity = range_doppler_validate(
        beat_signal, params, target_range
    )

    true_relative_velocity = target_velocity - drone_velocity
    print(f"True relative velocity: {true_relative_velocity:.2f} m/s")
    print(f"Range error:            {abs(est_range - target_range):.3f} m")
    print(f"Velocity error:         {abs(est_velocity - true_relative_velocity):.3f} m/s")

    # --- Plot: range-Doppler heatmap ---
    fig, ax = plt.subplots(figsize=(7, 5))
    extent = [velocities.min(), velocities.max(), ranges.min(), ranges.max()]
    im = ax.imshow(mag, aspect="auto", origin="lower", extent=extent, cmap="viridis")
    ax.set_xlabel("Relative velocity (m/s)")
    ax.set_ylabel("Range (m)")
    ax.set_title("Range-Doppler map (clean signal)")
    ax.axhline(target_range, color="r", linestyle="--", alpha=0.6, label="Ground truth range")
    ax.axvline(true_relative_velocity, color="orange", linestyle="--", alpha=0.6, label="Ground truth velocity")
    ax.set_ylim(0, Rmax)
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, label="Magnitude")

    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/range_doppler_map.png", dpi=150)
    print("\nSaved plot to outputs/range_doppler_map.png")