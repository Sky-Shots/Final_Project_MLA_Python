# --------------------------------------------------------------------
# A4 – Frame-Based Frequency Tracking
#
# Description:
# This file analyses the MLA waveform in smaller consecutive frames
# instead of processing the entire signal at once. For each frame,
# the DC offset is removed, a Hann window is applied, and an FFT is
# computed. The strongest FFT peak within each frame represents the
# resonance frequency during that time segment.
#
# The output is a sequence of peak frequencies over time, which makes
# it possible to observe:
#   - gradual drift in resonance (temperature, loading effects)
#   - sudden jumps caused by noise or clipping
#   - frames where the peak weakens or disappears
#   - “too perfect” behaviour indicating incorrect metadata or wrong
#     overtone selection
#
# A4 forms the basis of A5–A9. If time-resolved tracking is unstable,
# unrealistic, or flat, the later MLA steps will not produce valid
# results.
#
# Equations used:
#
#   DC removal:
#       x_clean[n] = x[n] − mean(x)
#
#   Hann window:
#       w[n] = 0.5 * (1 − cos(2πn / (N−1)))
#       x_win[n] = x_clean[n] * w[n]
#
#   FFT:
#       X[k] = Σ x_win[n] * exp(−j 2πkn / N)
#
#   Magnitude:
#       |X[k]| = sqrt( Re(X[k])² + Im(X[k])² )
#
#   Frame peak:
#       f_peak = k_peak * (fs / N)
#
# --------------------------------------------------------------------

import numpy as np
import json
import os
import sys
import csv
import matplotlib.pyplot as plt

print("\n--------------------------------------------------")
print("A4 –  Frame-Based Frequency Tracking")
print("--------------------------------------------------\n")


print("\nEquations used:")
print("x_clean[n] = x[n] - mean(x)")
print("w[n] = 0.5 * (1 - cos(2πn/(N-1)))")
print("x_win[n] = x_clean[n] * w[n]")
print("X[k] = Σ x_win[n] * exp(-j2πkn/N)")
print("|X[k]| = sqrt(Re(X[k])² + Im(X[k])²)")
print("f_peak = k_peak * (fs/N)")


RAW_FILE  = "mla_data.raw"
META_FILE = "mla_data.json"

# === DEFAULTS FOR SIMULATION ===
fs = 125_000_000    # only used if SIM_MODE or OLD_MODE
f0 = 15_000_000

# === MODE SWITCHES ===
SIM_MODE             = True      # synthetic drifting sine wave
NEW_PYTHON_FILE_MODE = False     # test_new.py format
OLD_PYTHON_FILE_MODE = False     # test.py format


# --- SIMULATION MODE ---

if SIM_MODE:

    M        = 25        # number of frames
    N_frame  = 8192      # samples per frame
    DRIFT_HZ = 50        # drift per frame

    print("SIM_MODE = TRUE → generating drifting sine wave...")

    peaks = []
    t_frame = np.arange(N_frame) / fs

    for m in range(M):
        f_m = f0 + m * DRIFT_HZ
        x_m = np.sin(2*np.pi*f_m*t_frame).astype(np.float64)

        w_m     = np.hanning(N_frame)
        X_m     = np.fft.rfft(x_m * w_m)
        freqs_m = np.fft.rfftfreq(N_frame, d=1.0/fs)
        mag_m   = np.abs(X_m) / (N_frame/2)

        k_m = np.argmax(mag_m)
        peaks.append(freqs_m[k_m])


# --- FILE MODE ---

else:

    # Error if both modes are False
    if not NEW_PYTHON_FILE_MODE and not OLD_PYTHON_FILE_MODE:
        raise ValueError("Enable NEW_PYTHON_FILE_MODE or OLD_PYTHON_FILE_MODE")

    # ---- OLD MLA FORMAT ---
    if OLD_PYTHON_FILE_MODE:
        print("OLD_PYTHON_FILE_MODE = TRUE → loading test.py MLA format")

        if not os.path.exists(RAW_FILE):
            print(f"ERROR: Missing {RAW_FILE}")
            sys.exit(1)

        with open(RAW_FILE, "rb") as f:
            x = np.frombuffer(f.read(), dtype=np.int16)

        fs = 125_000_000   # old MLA does not store metadata

    # --- NEW MLA FORMAT ---
    elif NEW_PYTHON_FILE_MODE:
        print("NEW_PYTHON_FILE_MODE = TRUE → loading test_new.py MLA format")

        if not os.path.exists(RAW_FILE) or not os.path.exists(META_FILE):
            print(f"ERROR: Missing {RAW_FILE} or {META_FILE}")
            sys.exit(1)

        with open(META_FILE, 'r') as f:
            meta = json.load(f)

        # get sample rate from metadata
        fs = float(meta.get("sample_rate_hz", fs))
        snapped = np.array(meta.get("snapped_freqs_hz", []), dtype=float)

        with open(RAW_FILE, "rb") as f:
            x = np.frombuffer(f.read(), dtype=np.int16)

        # Warnings for metadata
        if "snapped_freqs_hz" not in meta:
            print("[WARN] Metadata missing: snapped_freqs_hz")
            print("Fix:")
            print("  - Update test file to write snapped_freqs_hz")
            print("  - Confirm correct overtone selection before MLA run\n")

        if fs <= 0:
            print("[WARN] Invalid sample rate in metadata — using fallback.")
            

    # --- FRAME-BY-FRAME FFT ----

    N_frame = 8192
    M = x.size // N_frame

    if M < 1:
        print("[A4] ERROR: Not enough samples for even one frame.")
        sys.exit(1)

    w_m     = np.hanning(N_frame)
    freqs_m = np.fft.rfftfreq(N_frame, d=1.0/fs)

    peaks = []

    for m in range(M):
        start = m * N_frame
        stop  = start + N_frame

        x_m = x[start:stop].astype(np.float64)
        x_m = x_m - np.mean(x_m)

        X_m   = np.fft.rfft(x_m * w_m)
        mag_m = np.abs(X_m) / (N_frame/2)

        k_m = np.argmax(mag_m)
        peaks.append(freqs_m[k_m])


# --- WARNINGS ----

if np.std(peaks) < 1:
    print("[WARN] Tracking too flat — possible wrong tone or too perfect.")
    print("Fix:")
    print("  - Check overtone index in test.py/test_new.py")
    print("  - Verify JSON metadata contains snapped_freqs_hz")
    print("  - Increase frame size or MLA capture length")
    print("  - Refer to MLA document: Tracking Section\n")


if np.std(peaks) > 2e5:
    print("[WARN] Tracking very unstable — possible noise or clipping.")
    print("Fix:")
    print("  - Reduce MLA AMP value")
    print("  - Inspect TD0 jitter behaviour")
    print("  - Check grounding, balun connection, and cables")
    print("  - Ensure no clipping occurs in the MLA waveform\n")


# ---PLOTTING RAW TRACKING ---

plt.figure()
plt.plot(range(len(peaks)), np.array(peaks)/1e6, marker="o")
plt.xlabel("Frame number")
plt.ylabel("Peak frequency (MHz)")
plt.title("A4: Frequency Tracking")
plt.grid(True)
plt.tight_layout()
plt.savefig("a4_tracking.png", dpi=150)
print("saved: a4_tracking.png")
plt.show()


# --- SMOOTHED PLOTTING ---

SMOOTH_LEN = 5
smooth = np.convolve(peaks, np.ones(SMOOTH_LEN)/SMOOTH_LEN, mode="same")

plt.figure()
plt.plot(range(len(peaks)), np.array(peaks)/1e6, "o-", label="raw")
plt.plot(range(len(peaks)), smooth/1e6, "r-", linewidth=2, label="smoothed")
plt.xlabel("Frame number")
plt.ylabel("Peak frequency (MHz)")
plt.title("A4: Frequency Tracking (Smoothed)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("a4_tracking_smoothed.png", dpi=150)
print("saved: a4_tracking_smoothed.png")
plt.show()


