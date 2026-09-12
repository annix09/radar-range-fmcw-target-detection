"""
FMCW radar signal generation for drone radar project.
Person A deliverable: clean chirp -> beat signal -> range FFT, no noise/clutter.
Interface intended to hand off a range estimate to downstream (Doppler/CFAR) stage.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Radar / waveform parameters (from Radar Designer sizing)
# ---------------------------------------------------------------------------
c = 3e8            # speed of light (m/s)
fc = 24e9          # center frequency (Hz)
B = 150e6          # sweep bandwidth (Hz)
Tsweep = 10e-6      # sweep (chirp) duration (s)
Fs = 50e6          # sample rate (Hz)

# Derived design metrics (sanity check against Radar Designer output)
range_resolution = c / (2 * B)
range_max = (Fs * c * Tsweep) / (4 * B)

print(f"Range resolution (dR): {range_resolution:.2f} m")
print(f"Max unambiguous range (R_max): {range_max:.2f} m")

# ---------------------------------------------------------------------------
# Core function: generate a clean beat signal + range estimate for one target
# ---------------------------------------------------------------------------
def generate_beat_signal(target_range_m, fc=fc, B=B, Tsweep=Tsweep, Fs=Fs, c=c):
    """
    Simulate a single clean point-target FMCW return (no noise/clutter).

    Parameters
    ----------
    target_range_m : float
        Ground-truth range to the simulated target (m).

    Returns
    -------
    beat_signal : np.ndarray (complex)
        The dechirped beat signal for one sweep.
    t : np.ndarray
        Time vector for the sweep (s).
    """
    n_samples = int(Fs * Tsweep)
    t = np.linspace(0, Tsweep, n_samples, endpoint=False)

    k = B / Tsweep  # chirp rate (Hz/s)

    # Transmit chirp (up-chirp, sawtooth FMCW)
    tx = np.exp(1j * 2 * np.pi * (fc * t + 0.5 * k * t**2))

    # Round-trip delay for the target
    tau = 2 * target_range_m / c

    # Received chirp: same waveform, time-shifted by tau.
    # Using the same time base with a phase-delayed model (valid since tau << Tsweep).
    rx = np.exp(1j * 2 * np.pi * (fc * (t - tau) + 0.5 * k * (t - tau) ** 2))

    # Dechirp: mix tx with conjugate of rx -> beat signal at frequency k*tau
    beat_signal = tx * np.conj(rx)

    return beat_signal, t


def estimate_range(beat_signal, Fs=Fs, B=B, Tsweep=Tsweep, c=c):
    """
    Range FFT: take the beat signal, find peak frequency, convert to range.
    """
    n = len(beat_signal)
    window = np.hanning(n)
    spectrum = np.fft.fft(beat_signal * window)
    freqs = np.fft.fftfreq(n, d=1 / Fs)

    # Only positive frequencies matter for an up-chirp beat tone
    half = n // 2
    mag = np.abs(spectrum[:half])
    freqs_pos = freqs[:half]

    peak_idx = np.argmax(mag)
    beat_freq = freqs_pos[peak_idx]

    est_range = (beat_freq * c * Tsweep) / (2 * B)

    return est_range, freqs_pos, mag


# ---------------------------------------------------------------------------
# Validation run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ground_truth_range = 120.0  # meters, pick any value < range_max

    beat_signal, t = generate_beat_signal(ground_truth_range)
    est_range, freqs_pos, mag = estimate_range(beat_signal)

    print(f"\nGround-truth range: {ground_truth_range:.2f} m")
    print(f"Estimated range:    {est_range:.2f} m")
    print(f"Error:              {abs(est_range - ground_truth_range):.3f} m "
          f"(1 range bin = {range_resolution:.2f} m)")

    # Plot: beat signal (time domain) and range spectrum
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(t * 1e6, np.real(beat_signal))
    axes[0].set_title("Beat signal (real part)")
    axes[0].set_xlabel("Time (us)")
    axes[0].set_ylabel("Amplitude")

    ranges = (freqs_pos * c * Tsweep) / (2 * B)
    axes[1].plot(ranges, mag)
    axes[1].axvline(ground_truth_range, color="r", linestyle="--", label="Ground truth")
    axes[1].set_title("Range profile (FFT)")
    axes[1].set_xlabel("Range (m)")
    axes[1].set_ylabel("Magnitude")
    axes[1].set_xlim(0, range_max)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("fmcw_range_validation.png", dpi=150)
    print("\nSaved plot to fmcw_range_validation.png")