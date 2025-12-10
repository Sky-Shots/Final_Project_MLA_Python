import numpy as np
import matplotlib.pyplot as plt

# --- System Constants ---
SAMPLING_RATE = 125_000_000  # 125 MHz
SAMPLE_WIDTH = 16
SAMPLE_SIZE_BYTES = SAMPLE_WIDTH // 8 # Should be 2 bytes

DAC_WIDTH = 14
FULL_SCALE_VOLTAGE = 2  # Volts
SCALE_FACTOR = FULL_SCALE_VOLTAGE / (2**DAC_WIDTH)

MAX_MEMORY_MB = 64 # Example limit, adjust as needed for actual hardware
MAX_MEMORY_BYTES = MAX_MEMORY_MB * 1024 * 1024
MAX_SAMPLES_THEORETICAL = MAX_MEMORY_BYTES // SAMPLE_SIZE_BYTES # Theoretical max based on RAM

MAX_FREQ_COMPONENTS = 128 # Software/firmware limit on number of tones

# --- Constraint Definition ---
REQUIRED_BYTES_MULTIPLE = 64 # The total bytes must be a multiple of this

# --- Derived Constraint ---
# Ns * SAMPLE_SIZE_BYTES must be a multiple of REQUIRED_BYTES_MULTIPLE
# Ns * 2 must be a multiple of 64
# Ns must be a multiple of 64 / 2 = 32
REQUIRED_NS_MULTIPLE = REQUIRED_BYTES_MULTIPLE // SAMPLE_SIZE_BYTES
if REQUIRED_BYTES_MULTIPLE % SAMPLE_SIZE_BYTES != 0:
    raise ValueError(f"REQUIRED_BYTES_MULTIPLE ({REQUIRED_BYTES_MULTIPLE}) must be divisible by SAMPLE_SIZE_BYTES ({SAMPLE_SIZE_BYTES})")


# --- Tuning Function ---
def calculate_tuned_params(
    desired_freqs: list[float] | np.ndarray,
    desired_bw: float,
    sample_rate: float = SAMPLING_RATE,
    required_ns_multiple: int = REQUIRED_NS_MULTIPLE,
    max_samples_limit: int = MAX_SAMPLES_THEORETICAL
    ) -> tuple[np.ndarray, float, int, np.ndarray]:
    """
    Calculates the closest achievable frequencies and bandwidth
    satisfying the lock-in and hardware memory alignment constraints.

    The primary hardware constraint is that the number of samples (Ns)
    must be an integer multiple of `required_ns_multiple`.

    Args:
        desired_freqs (list or np.array): List of desired frequencies in Hz.
        desired_bw (float): Desired measurement bandwidth (base tone spacing) in Hz.
        sample_rate (float): System sample rate in Hz.
        required_ns_multiple (int): Ns must be an integer multiple of this value
                                   (derived from byte alignment needs).
        max_samples_limit (int): Maximum allowed number of samples (Ns).

    Returns:
        tuple: (tuned_freqs, tuned_bw, Ns_actual, n_actual_list)
            - tuned_freqs (np.array): Array of achievable frequencies in Hz.
            - tuned_bw (float): Achievable measurement bandwidth / base tone spacing in Hz.
            - Ns_actual (int): The number of samples per measurement window (pixel/period).
            - n_actual_list (np.array): Array of integers n_i such that tuned_freqs[i] = n_i * tuned_bw.

    Raises:
        ValueError: If desired_bw is not positive.
        ValueError: If max_samples_limit is too small for required_ns_multiple.
    """
    if desired_bw <= 0:
        raise ValueError("Desired bandwidth must be positive.")
    if required_ns_multiple <= 0:
         raise ValueError("required_ns_multiple must be positive.")
    if max_samples_limit < required_ns_multiple:
         raise ValueError(f"max_samples_limit ({max_samples_limit}) is too small for required_ns_multiple ({required_ns_multiple})")


    # 1. Find Target Ns based on desired bandwidth (base tone spacing)
    # Ns = Fs / df => Target Ns = Fs / desired_bw
    Ns_target = sample_rate / desired_bw
    if Ns_target > max_samples_limit:
         print(f"Warning: Target Ns ({Ns_target:.0f}) exceeds max_samples_limit ({max_samples_limit}). "
               f"Resulting Ns will be capped.")

    # 2. Find Closest Allowed Ns (multiple of required_ns_multiple)
    # k = Ns / required_ns_multiple
    k_target = Ns_target / required_ns_multiple
    k_actual = round(k_target)
    # Ensure Ns is at least required_ns_multiple
    if k_actual < 1:
        k_actual = 1
    Ns_actual = int(k_actual * required_ns_multiple) # Ensure integer type

    # 3. Cap Ns at maximum allowed samples if necessary
    if Ns_actual > max_samples_limit:
        print(f"Warning: Calculated Ns ({Ns_actual}) exceeds max_samples_limit ({max_samples_limit}).")
        # Find the largest multiple of required_ns_multiple <= max_samples_limit
        Ns_actual = (max_samples_limit // required_ns_multiple) * required_ns_multiple
        if Ns_actual == 0: # Should not happen based on initial check, but safety first
             raise ValueError(f"Cannot find valid Ns multiple within max_samples_limit.")
        print(f"Adjusting Ns to {Ns_actual}.")
        k_actual = Ns_actual // required_ns_multiple


    # 4. Calculate Actual Tuned Bandwidth (base tone spacing)
    # df = Fs / Ns
    tuned_bw = sample_rate / Ns_actual

    # 5. Calculate Actual Tuned Frequencies
    # f_i = n_i * df
    tuned_freqs = []
    n_actual_list = []
    unique_desired_freqs = sorted(list(set(desired_freqs))) # Process unique frequencies

    for f_des in unique_desired_freqs:
        # Find closest integer multiple n_i for the desired frequency
        # n_i = f_des / df
        n_target = f_des / tuned_bw
        n_actual = round(n_target)

        # Ensure n_actual is positive if f_des > 0 (avoid DC unless requested)
        # Also handle potential rounding to 0 for very low frequencies near DC
        if f_des > 0 and n_actual <= 0:
             n_actual = 1 # Force to the first multiple of df if desired freq was positive

        f_actual = n_actual * tuned_bw

        # Prevent adding duplicate frequencies due to rounding
        # Check against already added *actual* frequencies using a tolerance
        is_duplicate = False
        tolerance = tuned_bw * 0.1 # Define a tolerance (e.g., 10% of df)
        for existing_f_actual in tuned_freqs:
            if abs(f_actual - existing_f_actual) < tolerance:
                # Check if they map to the same integer n
                existing_n = round(existing_f_actual / tuned_bw)
                if n_actual == existing_n:
                    is_duplicate = True
                    # print(f"Debug: Skipping duplicate freq {f_actual:.2f} (n={n_actual}), already have {existing_f_actual:.2f} (n={existing_n})")
                    break
        if not is_duplicate:
             tuned_freqs.append(f_actual)
             n_actual_list.append(int(n_actual))


    # Sort results for consistency
    if tuned_freqs: # Ensure lists are not empty before sorting
        results = sorted(zip(tuned_freqs, n_actual_list))
        tuned_freqs = np.array([f for f, n in results])
        n_actual_list = np.array([n for f, n in results])
    else:
        tuned_freqs = np.array([])
        n_actual_list = np.array([])


    return tuned_freqs, tuned_bw, Ns_actual, n_actual_list

# --- Helper for Generating Target Frequencies (Unchanged) ---
def generate_all_target_frequencies(center_frequencies: list[float], desired_bandwidth: float, comb_size: int) -> list[float]:
    """
    Generates a list of all desired frequencies including center and sidebands.

    Args:
        center_frequencies (list[float]): List of desired center frequencies in Hz.
        desired_bandwidth (float): The *initial* desired spacing for sidebands in Hz.
        comb_size (int): Total number of frequencies per center freq (must be odd).

    Returns:
        list[float]: A list containing all target frequencies.
    """
    if comb_size % 2 == 0:
        raise ValueError("Comb size must be odd.")

    all_freqs = set() # Use a set to automatically handle duplicates
    steps = (comb_size - 1) // 2

    for f_center in center_frequencies:
        all_freqs.add(f_center) # Add center frequency
        for i in range(1, steps + 1):
            f_upper = f_center + i * desired_bandwidth
            f_lower = f_center - i * desired_bandwidth
            if f_upper >= 0: # Only add non-negative frequencies
                all_freqs.add(f_upper)
            if f_lower >= 0: # Only add non-negative frequencies
                all_freqs.add(f_lower)

    return sorted(list(all_freqs))


# --- Main Waveform Generation Function ---
def freq_comb_twos_comp_dac(
    center_frequencies: list[float],
    desired_bandwidth: float,
    comb_size: int,
    amplitude_v: float,
    apply_t_shift: bool = True, # Option to center waveform phase
) -> tuple[np.ndarray, np.ndarray, float, int, np.ndarray]:
    """
    Generate a frequency comb waveform in two's complement format for the DAC,
    ensuring all frequencies are integer multiples of the tuned bandwidth (df),
    and the number of samples (Ns) is a multiple of REQUIRED_NS_MULTIPLE (e.g., 32),
    which ensures the total byte count is a multiple of REQUIRED_BYTES_MULTIPLE (e.g., 64).

    Args:
        center_frequencies (list[float]): List of *desired* center frequencies in Hz.
        desired_bandwidth (float): *Desired* bandwidth/spacing of the waveform in Hz.
        comb_size (int): Size of the frequency comb around each center (must be odd).
        amplitude_v (float): Desired *peak* voltage amplitude of the *combined* waveform.
        apply_t_shift (bool): If True, shifts the time vector to center the phases.

    Raises:
        ValueError: If comb_size is even.
        ValueError: If the total number of frequency components exceeds MAX_FREQ_COMPONENTS.
        ValueError: If constraints cannot be met (e.g., memory limit too low).

    Returns:
        tuple: (waveform_twos_comp, waveform_t, tuned_bw, Ns, tuned_freqs_list)
            - waveform_twos_comp (np.ndarray): Comb waveform in int16 two's complement.
            - waveform_t (np.ndarray): Time vector for one period of the waveform.
            - tuned_bw (float): The actual, tuned bandwidth/spacing used.
            - Ns (int): The actual number of samples used per period.
            - tuned_freqs_list (np.ndarray): List of the actual frequencies generated.
    """
    if comb_size % 2 == 0:
        raise ValueError("Comb size must be odd.")

    # 1. Generate all *target* frequencies based on desired inputs
    all_target_freqs = generate_all_target_frequencies(center_frequencies, desired_bandwidth, comb_size)

    if len(all_target_freqs) > MAX_FREQ_COMPONENTS:
        raise ValueError(f"Requested {len(all_target_freqs)} frequency components, exceeds limit of {MAX_FREQ_COMPONENTS}.")

    # 2. Tune *all* target frequencies and the bandwidth simultaneously
    # Pass the specific hardware constraint REQUIRED_NS_MULTIPLE
    tuned_freqs_list, tuned_bw, Ns, n_list = calculate_tuned_params(
        all_target_freqs,
        desired_bandwidth,
        sample_rate=SAMPLING_RATE,
        required_ns_multiple=REQUIRED_NS_MULTIPLE, # Use the derived value (32)
        max_samples_limit=MAX_SAMPLES_THEORETICAL
    )

    # Verify byte count constraint
    total_bytes = Ns * SAMPLE_SIZE_BYTES
    if total_bytes % REQUIRED_BYTES_MULTIPLE != 0:
         # This should ideally never happen if calculate_tuned_params is correct
         raise RuntimeError(f"Internal Error: Final Ns ({Ns}) does not satisfy byte multiple constraint! "
                            f"Total bytes: {total_bytes}, Required multiple: {REQUIRED_BYTES_MULTIPLE}")


    print(f"Desired Bandwidth: {desired_bandwidth:.2f} Hz")
    print(f"Tuned Bandwidth (df): {tuned_bw:.5f} Hz")
    print(f"Ns (Samples per period, multiple of {REQUIRED_NS_MULTIPLE}): {Ns}")
    print(f"Total Bytes (Ns * {SAMPLE_SIZE_BYTES}, multiple of {REQUIRED_BYTES_MULTIPLE}): {total_bytes}")
    print(f"Total unique tuned frequencies: {len(tuned_freqs_list)}")
    # print(f"Tuned Frequencies (Hz): {tuned_freqs_list}")
    # print(f"Corresponding Integers (n_i = f_i/df): {n_list}")

    # 3. Generate Time Vector based on Tuned Parameters
    waveform_duration = Ns / SAMPLING_RATE
    waveform_t = np.linspace(0, waveform_duration, Ns, endpoint=False)
    t_shift = waveform_duration / 2 if apply_t_shift else 0

    np.save("t_shift.npy", t_shift)
    print(f"Saved t_shift to file. t_shift.npy")
    np.save("freqs.npy", tuned_freqs_list)
    print(f"Saved tuned frequencies to file. freqs.npy")
    np.save("waveform_t.npy", waveform_t)
    print(f"Saved waveform time vector to file. waveform_t.npy")

    print(f"Actual waveform duration: {waveform_duration*1000:.6f} ms")

    # 4. Generate the Waveform by Summing Tuned Frequencies
    waveform_sum = np.zeros(Ns, dtype=np.float64)
    for freq in tuned_freqs_list:
        waveform_sum += np.sin(2 * np.pi * freq * (waveform_t - t_shift))

    # 5. Normalize and Scale
    waveform_normalized = waveform_sum / np.max(np.abs(waveform_sum))
    del waveform_sum
    waveform_volts = waveform_normalized * amplitude_v

    # 6. Convert to Two's Complement for DAC
    waveform_dac_scaled = np.round(waveform_volts / SCALE_FACTOR)
    del waveform_volts
    waveform_twos_comp = waveform_dac_scaled.astype(np.int16)

    return waveform_twos_comp, waveform_t, tuned_bw, Ns, tuned_freqs_list


# --- Example Usage (Unchanged from previous, but reflects new internal logic) ---
if __name__ == "__main__":
    print(f"--- System Configuration ---")
    print(f"Sample Rate: {SAMPLING_RATE/1e6} MHz")
    print(f"Sample Size: {SAMPLE_SIZE_BYTES} bytes")
    print(f"Required Total Bytes Multiple: {REQUIRED_BYTES_MULTIPLE}")
    print(f"--> Required Ns Multiple: {REQUIRED_NS_MULTIPLE}")
    print(f"Max Samples (Theoretical): {MAX_SAMPLES_THEORETICAL}")
    print(f"---------------------------")

    # --- Parameters ---
    center_freqs_des = [4.95e6, 14.85e6, 24.75e6, 34.65e6, 44.55e6]
    desired_bw_des = 500.0 # Hz
    comb_sz = 11
    amplitude_pk_v = 0.3 # Volts

    # --- Generate Waveform ---
    try:
        comb_waveform_tc, t_vec, actual_bw, n_samples, actual_freqs = freq_comb_twos_comp_dac(
            center_freqs_des,
            desired_bw_des,
            comb_sz,
            amplitude_pk_v,
            apply_t_shift=True
        )

        print(f"\nGenerated waveform with {n_samples} samples.")
        print(f"Actual frequencies generated (MHz): \n{actual_freqs / 1e6}")

        # --- Analysis & Plotting ---
        # Plot the waveform
        plt.figure(figsize=(12, 6))
        plt.plot(t_vec * 1e6, comb_waveform_tc / 8192) # Convert to volts
        plt.title("Frequency Comb Waveform")
        plt.xlabel("Time (us)")
        plt.ylabel("Amplitude (V)")
        plt.grid()
        plt.ylim(-1.5, 1.5) # Adjust y-limits for better visibility

        plt.show()
        

    except ValueError as e:
        print(f"Error generating waveform: {e}")
    except Exception as e: # Catch other potential errors
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()