import numpy as np
import pandas as pd

from signalgen import params, generate_clean_signal




def calculate_range_history(
    target_range,
    target_velocity,
    drone_velocity,
    params
):
    """
    Calculate target range at the beginning of each chirp.

    Positive relative velocity = target moving away
    from the drone.
    """

    Tc = params["Tc"]
    N = params["N"]

    # Relative radial velocity
    relative_velocity = target_velocity - drone_velocity

    # Time at beginning of each chirp
    chirp_times = np.arange(N) * Tc

    # Range changes with time
    range_history = (
        target_range +
        relative_velocity * chirp_times
    )

    return range_history


# ============================================================
# TASK 2
# RECEIVER NOISE
# ============================================================

def add_receiver_noise(signal, snr_db):
    """
    Add complex Gaussian receiver noise.

    signal : complex radar signal
    snr_db : desired SNR in dB
    """

    # Average signal power
    signal_power = np.mean(np.abs(signal) ** 2)

    # Convert SNR from dB to linear
    snr_linear = 10 ** (snr_db / 10)

    # Required noise power
    noise_power = signal_power / snr_linear

    # Generate complex Gaussian noise
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(*signal.shape)
        + 1j * np.random.randn(*signal.shape)
    )

    # Add noise
    noisy_signal = signal + noise

    return noisy_signal


# ============================================================
# TASK 3
# GROUND CLUTTER
# ============================================================

def add_ground_clutter(signal, clutter_power):
    """
    Add simulated ground clutter.

    This is a simplified Python clutter model.
    """

    # Generate complex clutter
    clutter = np.sqrt(clutter_power / 2) * (
        np.random.randn(*signal.shape)
        + 1j * np.random.randn(*signal.shape)
    )

    # Add clutter to received signal
    cluttered_signal = signal + clutter

    return cluttered_signal


# ============================================================
# TASK 4
# RANGE-DOPPLER PROCESSING
# ============================================================

def range_doppler_processing(signal, params):
    """
    Perform:
        1. Range FFT
        2. Doppler FFT

    Input:
        signal = [Nsamples x Nchirps]

    Output:
        ranges
        velocities
        range_doppler
    """

    fc = params["fc"]
    B = params["B"]
    Tc = params["Tc"]
    Fs = params["Fs"]
    N = params["N"]

    c = 3e8

    n_samples = signal.shape[0]

    # --------------------------------------------------------
    # RANGE FFT
    # --------------------------------------------------------

    # Window across samples
    range_window = np.hanning(n_samples)[:, None]

    windowed_signal = signal * range_window

    # FFT along fast-time/sample dimension
    range_fft = np.fft.fft(
        windowed_signal,
        axis=0
    )

    # Only positive frequencies
    half = n_samples // 2

    range_fft = range_fft[:half, :]

    # Frequency associated with each range bin
    frequencies = np.fft.fftfreq(
        n_samples,
        d=1 / Fs
    )[:half]

    # Beat frequency -> range
    ranges = (
        frequencies *
        c *
        Tc /
        (2 * B)
    )

    # --------------------------------------------------------
    # DOPPLER FFT
    # --------------------------------------------------------

    # Window across chirps
    doppler_window = np.hanning(N)[None, :]

    windowed_range_fft = (
        range_fft *
        doppler_window
    )

    # FFT across chirps
    range_doppler = np.fft.fftshift(
        np.fft.fft(
            windowed_range_fft,
            axis=1
        ),
        axes=1
    )

    # Doppler frequencies
    doppler_frequencies = np.fft.fftshift(
        np.fft.fftfreq(
            N,
            d=Tc
        )
    )

    # Radar wavelength
    wavelength = c / fc

    # Doppler frequency -> radial velocity
    velocities = (
        doppler_frequencies *
        wavelength /
        2
    )

    return ranges, velocities, range_doppler


# ============================================================
# TASK 4
# MTI / DOPPLER CLUTTER REJECTION
# ============================================================

def apply_mti(range_doppler, velocities, clutter_velocity=0.0):
    """
    Suppress Doppler bins close to zero velocity.

    This is a simple zero-Doppler rejection filter.
    """

    filtered = range_doppler.copy()

    # Width of zero-Doppler region to remove
    rejection_width = 0.15  # m/s

    # Find bins close to zero velocity
    clutter_bins = (
        np.abs(velocities - clutter_velocity)
        < rejection_width
    )

    # Suppress those bins
    filtered[:, clutter_bins] = 0

    return filtered


# ============================================================
# TASK 5
# CA-CFAR
# ============================================================
def ca_cfar(
    range_profile,
    num_training_cells=10,
    num_guard_cells=2,
    threshold_factor=3.0  # Tuned for better detection at low SNR
):

    """
    1-D Cell Averaging CFAR.

    Returns:
        detections
        threshold
    """

    power = np.abs(range_profile) ** 2

    N = len(power)

    detections = np.zeros(N, dtype=bool)

    threshold = np.zeros(N)

    for i in range(
        num_training_cells + num_guard_cells,
        N - num_training_cells - num_guard_cells
    ):

        # Training cells on left
        left_start = (
            i -
            num_guard_cells -
            num_training_cells
        )

        left_end = (
            i -
            num_guard_cells
        )

        left_cells = power[
            left_start:left_end
        ]

        # Training cells on right
        right_start = (
            i +
            num_guard_cells +
            1
        )

        right_end = (
            right_start +
            num_training_cells
        )

        right_cells = power[
            right_start:right_end
        ]

        # Combine training cells
        training_cells = np.concatenate(
            [left_cells, right_cells]
        )

        # Estimate local noise power
        noise_estimate = np.mean(
            training_cells
        )

        # CFAR threshold
        threshold[i] = (
            threshold_factor *
            noise_estimate
        )

        # Cell under test
        if power[i] > threshold[i]:
            detections[i] = True

    return detections, threshold


# ============================================================
# TASK 5
# FIND TARGET RANGE
# ============================================================

def find_detected_range(
    range_profile,
    ranges,
    detections
):
    """
    Find strongest CA-CFAR detection.
    """

    detected_indices = np.where(
        detections
    )[0]

    if len(detected_indices) == 0:
        return np.nan

    # Power at detected bins
    powers = np.abs(
        range_profile[detected_indices]
    ) ** 2

    # Strongest detection
    strongest_index = (
        detected_indices[
            np.argmax(powers)
        ]
    )

    return ranges[strongest_index]


# ============================================================

# ============================================================

def detectRange(
    beatSignal,
    params,
    snr_db=10,
    clutter_power=0.1
):
   
    # --------------------------------------------------------
    # TASK 2: Add receiver noise
    # --------------------------------------------------------

    noisy_signal = add_receiver_noise(
        beatSignal,
        snr_db
    )

    # --------------------------------------------------------
    # TASK 3: Add ground clutter
    # --------------------------------------------------------

    cluttered_signal = add_ground_clutter(
        noisy_signal,
        clutter_power
    )

    # --------------------------------------------------------
    # TASK 4: Range-Doppler processing
    # --------------------------------------------------------

    ranges, velocities, range_doppler = (
        range_doppler_processing(
            cluttered_signal,
            params
        )
    )

    # --------------------------------------------------------
    # TASK 4: MTI
    # --------------------------------------------------------

    filtered_range_doppler = apply_mti(
        range_doppler,
        velocities
    )

    # --------------------------------------------------------
    # Convert range-Doppler map to range profile
    #
    # Take strongest Doppler response for each range bin.
    # --------------------------------------------------------

    range_profile = np.max(
        np.abs(filtered_range_doppler),
        axis=1
    )

    # --------------------------------------------------------
    # TASK 5: CA-CFAR
    # --------------------------------------------------------

    detections, threshold = ca_cfar(
        range_profile
    )

    # --------------------------------------------------------
    # Find detected range
    # --------------------------------------------------------

    measured_range = find_detected_range(
        range_profile,
        ranges,
        detections
    )

    detected = not np.isnan(
        measured_range
    )

    # --------------------------------------------------------
    # Estimate SNR at detected peak
    # --------------------------------------------------------

    if detected:

        detected_index = np.argmin(
            np.abs(
                ranges -
                measured_range
            )
        )

       
        signal_power = (
            range_profile[detected_index]
        )

        # Estimate noise from non-detection bins
        noise_bins = range_profile[
            ~detections
        ]

        if len(noise_bins) > 0:

            noise_power = np.mean(
                noise_bins
            )

        else:

            noise_power = 1e-12

        estimated_snr = 20 * np.log10(
            signal_power /
            (noise_power + 1e-12)
        )

    else:

        estimated_snr = np.nan

    # --------------------------------------------------------
    # Required output format
    # --------------------------------------------------------

    detection = {
        "frameTime": 0.0,
        "detected": detected,
        "measuredRange": measured_range,
        "snr_dB": estimated_snr
    }

    return detection


# ============================================================

# ============================================================

if __name__ == "__main__":

    print("\n========================================")
    print(" FMCW DRONE RADAR - PERSON B")
    print("========================================")

    # --------------------------------------------------------
    # Target parameters
    # --------------------------------------------------------

    target_range = 120.0
    target_velocity = 8.0
    drone_velocity = 0.0

    # --------------------------------------------------------
    # TASK 1
    # --------------------------------------------------------

    print("\nTASK 1: MOTION")

    range_history = calculate_range_history(
        target_range,
        target_velocity,
        drone_velocity,
        params
    )

    relative_velocity = (
        target_velocity -
        drone_velocity
    )

    print(
        f"Relative velocity: "
        f"{relative_velocity:.2f} m/s"
    )

    print(
        f"Initial range: "
        f"{range_history[0]:.5f} m"
    )

    print(
        f"Final range: "
        f"{range_history[-1]:.5f} m"
    )

    # --------------------------------------------------------

    # --------------------------------------------------------

    beat_signal = generate_clean_signal(
        target_range,
        target_velocity,
        drone_velocity,
        params
    )

    print(
        f"\nBeat signal shape: "
        f"{beat_signal.shape}"
    )

    # --------------------------------------------------------
    # TASK 2
    # --------------------------------------------------------

    print("\nTASK 2: RECEIVER NOISE")

    snr_db = 10

    noisy_signal = add_receiver_noise(
        beat_signal,
        snr_db
    )

    print(
        f"SNR setting: "
        f"{snr_db} dB"
    )

    print(
        f"Noisy signal shape: "
        f"{noisy_signal.shape}"
    )

    # --------------------------------------------------------
    # TASK 3
    # --------------------------------------------------------

    print("\nTASK 3: GROUND CLUTTER")

    clutter_power = 0.1

    cluttered_signal = add_ground_clutter(
        noisy_signal,
        clutter_power
    )

    print(
        f"Clutter power: "
        f"{clutter_power}"
    )

    print(
        f"Cluttered signal shape: "
        f"{cluttered_signal.shape}"
    )

    # --------------------------------------------------------
    # TASK 4
    # --------------------------------------------------------

    print("\nTASK 4: DOPPLER + MTI")

    ranges, velocities, range_doppler = (
        range_doppler_processing(
            cluttered_signal,
            params
        )
    )

    filtered_rd = apply_mti(
        range_doppler,
        velocities
    )

    print(
        f"Range bins: "
        f"{len(ranges)}"
    )

    print(
        f"Doppler bins: "
        f"{len(velocities)}"
    )

    print("MTI filtering completed.")

    # --------------------------------------------------------
    # TASK 5
    # --------------------------------------------------------

    print("\nTASK 5: CA-CFAR")

    range_profile = np.max(
        np.abs(filtered_rd),
        axis=1
    )

    detections, threshold = ca_cfar(
        range_profile
    )

    detected_range = find_detected_range(
        range_profile,
        ranges,
        detections
    )

    if np.isnan(detected_range):

        print("No target detected.")

    else:

        print(
            f"Detected range: "
            f"{detected_range:.2f} m"
        )

    # --------------------------------------------------------
    # COMPLETE DETECTION FUNCTION
    # --------------------------------------------------------

    print("\n========================================")
    print(" FINAL DETECTION OUTPUT")
    print("========================================")

    detection = detectRange(
        beat_signal,
        params,
        snr_db=10,
        clutter_power=0.1
    )

    print(
        f"Frame time:     "
        f"{detection['frameTime']:.3f} s"
    )

    print(
        f"Detected:       "
        f"{detection['detected']}"
    )

    print(
        f"Measured range: "
        f"{detection['measuredRange']}"
    )

    print(
        f"Estimated SNR:  "
        f"{detection['snr_dB']}"
    )