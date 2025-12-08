# ============================================================
# A9_phase_view.py  (MLA – magnitude + phase quick-look)
#
# What this script does:
#   - Takes one MLA time-domain signal (x) and its sample rate (fs)
#   - Runs an FFT (frequency-domain analysis)
#   - Computes:
#       • magnitude spectrum  → mag, mag_db
#       • phase spectrum      → phase (angle of each FFT bin)
#   - This is a simple “quick look” tool to see both
#       • how strong the signal is at each frequency
#       • what the phase looks like at each frequency
#
# Inputs:
#   SIM_MODE = True:
#       - No files needed.
#       - Script creates a clean test sine:
#            fs  = 125 MHz      (sample rate)
#            N   = 50 000       (number of samples)
#            f0  = 15 MHz       (test tone)
#       - Result: x is a pure sine wave.
#
#   SIM_MODE = False:
#       - Needs these two files in the same folder:
#            RAW_FILE  = "mla_data.raw"   (int16 time samples from board)
#            META_FILE = "mla_data.json"  (JSON with sample_rate_hz, etc.)
#       - Reads:
#            fs  = meta["sample_rate_hz"]
#            x   = samples from mla_data.raw
#
# Outputs (in Python, not yet saved to disk):
#   - x        : 1D NumPy array with time-domain samples (float64)
#   - fs       : sample rate in Hz (float)
#   - freqs    : 1D NumPy array with FFT bin frequencies (Hz)
#   - mag      : magnitude of FFT bins (linear)
#   - mag_db   : magnitude in dB (20*log10)
#   - phase    : phase angle of each FFT bin (radians, -pi .. +pi)
#
#   - Prints a short summary to the terminal:
#       • which mode was used (SIM or FILE)
#       • fs, N
#       • FFT size and frequency range
#
# ------------------------------------------------------------
#  Equations used (printed + commented for my memory)
# ------------------------------------------------------------
#
#  FFT Magnitude:
#       |X(f)| = sqrt( Re(X)^2 + Im(X)^2 )
#
#  Phase Unwrapping:
#       φ_unwrap[n] = φ_raw[n] + 2π * k     (k removes big jumps)
#
#  Impedance approximation:
#       Z(f) ≈ 1 / |X(f)|
#
#  RMS (signal strength):
#       RMS = sqrt( (1/N) * Σ x[n]^2 )
#
#  Peak detection:
#       f_peak = f[argmax(|X(f)|)]
#
#  These are directly connected to the QCM analysis equations
#  in the thesis (Sauerbrey shift, Kanazawa–Gordon behaviour).
#
# ============================================================


import numpy as np
import json
import matplotlib.pyplot as plt
import sys
import os

print("\n--------------------------------------------------")
print("A9 -phase_view (MLA – magnitude + phase quick-look)")
print("--------------------------------------------------\n")
# ============================================================
# FIXED FILE MODES
# ============================================================
SIM_MODE             = True
NEW_PYTHON_FILE_MODE = False
OLD_PYTHON_FILE_MODE = False

RAW_FILE  = "mla_data.raw"
META_FILE = "mla_data.json"

# ------------------------------------------------------------
# STEP 0 – SAFE FILE LOADING (FIXED)
# ------------------------------------------------------------
if SIM_MODE:
    # Simulation mode (keep your own logic)
    fs = 125_000_000
    N  = 4096
    t  = np.arange(N) / fs
    f0 = 15_000_000
    x  = np.sin(2 * np.pi * f0 * t)

    print(f"[A9] SIM_MODE = TRUE | fs={fs} | f0={f0} | N={N}")

else:
    if not NEW_PYTHON_FILE_MODE and not OLD_PYTHON_FILE_MODE:
        raise ValueError("Enable NEW or OLD MLA mode!")

    # ---- NEW MLA FORMAT ----
    if NEW_PYTHON_FILE_MODE:
        print("[A9] NEW MLA MODE (RAW+JSON)")

        if not os.path.exists(RAW_FILE):
            sys.exit("ERROR: RAW file missing.")

        # Load metadata if available
        if os.path.exists(META_FILE):
            with open(META_FILE, "r") as f:
                meta = json.load(f)
            fs = float(meta.get("sample_rate_hz", 125_000_000))
        else:
            print("[WARN] Missing metadata → fs=125 MHz fallback")
            fs = 125_000_000

        with open(RAW_FILE, "rb") as f:
            x = np.frombuffer(f.read(), dtype=np.int16).astype(np.float64)

        x = x - np.mean(x)

    # ---- OLD MLA FORMAT ----
    elif OLD_PYTHON_FILE_MODE:
        print("[A9] OLD MLA MODE (RAW only)")

        if not os.path.exists(RAW_FILE):
            sys.exit("ERROR: RAW file missing.")

        with open(RAW_FILE, "rb") as f:
            x = np.frombuffer(f.read(), dtype=np.int16).astype(np.float64)

        # No metadata → default fs
        fs = 125_000_000
        x  = x - np.mean(x)

print(f"[A9] Loaded {len(x)} samples | fs = {fs} Hz")


# Step 1 – Time‐domain stats
x = x.astype(np.float64)
x_zm = x - np.mean(x)
rms = np.sqrt(np.mean(x_zm**2))
p2p = np.max(x_zm) - np.min(x_zm)

print(f"Time-domain statistics:")
print(f"  RMS amplitude:      {rms:.4f}")
print(f"  Peak-to-peak:       {p2p:.4f}")

# Step 2 – Plot wave
plt.figure()
plt.plot(x_zm)
plt.title("A9: Time-domain signal")
plt.xlabel("Sample index")
plt.ylabel("Amplitude (zero-mean)")
plt.grid(True)
plt.savefig("a9_wave.png",dpi=150)
plt.show()

# Step 3 – Histogram
plt.figure()
plt.hist(x_zm, bins=100)
plt.title("A9: Histogram (ADC distribution)")
plt.xlabel("Amplitude")
plt.ylabel("Count")
plt.grid(True)
plt.savefig("a9_hist.png", dpi=150)
plt.show()

# (FFT/Phase will be added later manually by you)
print("[A9] NOTE: FFT, phase, impedance not implemented yet—only file mode fixed.")


# ---------- STEP A9-FFT: Compute FFT, Magnitude, Phase ----------
print("\n --- [A9]: Computing FFT, Magnitude, and Phase --- ")

# Hann window (same style as A3)
w = np.hanning(len(x))
x_win = x * w

# ------------------------------------------------------------
# FFT analysis
# ------------------------------------------------------------
X = np.fft.rfft(x_win)
freqs = np.fft.rfftfreq(len(x), d=1/fs)

# Magnitude
mag = np.abs(X) / (len(x) / 2)

# Phase (unwrap for smoothness)
phase = np.unwrap(np.angle(X))

# Peak index
k_peak = np.argmax(mag)
peak_hz = freqs[k_peak]

print(f"Peak Frequency (FFT-based): {peak_hz/1e6:.6f} MHz")

# ---------- STEP A9-WARNINGS: Basic Signal Health Checks ----------
print("\n --- [A9]: Signal Health Checks ---")

warnings = []
fix = []

# 1. Clipping check
if np.max(x) >= 0.99 or np.min(x) <= -0.99:
    warnings.append("WARNING: Signal is clipped. Reduce DAC amplitude.")
    fix.append("Fix: Lower AMP value in test_new.py/test.py until max amplitude stays within ±0.9.")

# 2. Weak RMS check
rms = np.sqrt(np.mean(x**2))
if rms < 0.01:
    warnings.append("WARNING: Signal is very weak (RMS < 0.01). Increase amplitude or check contacts.")
    fix.append("Fix: Increase AMP slightly; check sensor contact and tighten wiring.")

# 3. High RMS check (possible clipping soon)
if rms > 0.3:
    warnings.append("WARNING: RMS unusually high (>0.3). Clipping likely.")
    fix.append("Fix: Reduce AMP by 10–20% to avoid distortion.")

# 4. Peak check (if peak magnitude suspiciously small)
if mag[k_peak] < 1e-4:
    warnings.append("WARNING: Very small FFT peak. Possibly noise or wrong sampling rate.")
    fix.append("Fix: Verify sample_rate_hz in JSON; ensure correct MLA capture window.")
    


# 5. Phase instability check
phase_diff = np.max(np.abs(np.diff(phase)))
if phase_diff > 3:
    warnings.append("WARNING: Phase unstable (large jumps). Low SNR or noise present.")
    fix.append("Fix: Improve grounding, reduce external noise, or slightly increase AMP.")

# Print all warnings
if warnings:
    print("\n".join(warnings))
    print("\n".join(fix))
else:
    print("No warnings detected.")
    print("No fix detected.")
    
# ---------- STEP A9-CSV: Export Frequency, Magnitude, Phase ----------
print("\n --- [A9]: Saving FFT data to CSV ---")

import csv

CSV_NAME = "a9_fft_phase.csv"

with open(CSV_NAME, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["freq_hz", "magnitude", "phase_rad"])
    for i in range(len(freqs)):
        writer.writerow([freqs[i], mag[i], phase[i]])

print(f"Saved CSV file: {CSV_NAME}")

# ---------- STEP A9-PLOT MAG: Magnitude Spectrum ----------
print("\n ---[A9]: Plotting Magnitude Spectrum ---")

plt.figure(figsize=(10, 5))
plt.plot(freqs/1e6, mag, linewidth=1)
plt.title("A9: Magnitude Spectrum")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Magnitude")
plt.grid(True)

# mark the peak
plt.axvline(peak_hz/1e6, color='red', linestyle='--', label=f"Peak ~ {peak_hz/1e6:.6f} MHz")
plt.legend()

plt.tight_layout()
plt.savefig("a9_magnitude.png", dpi=150)
print("Saved: a9_magnitude.png")


# ---------- STEP A9-PLOT PHASE: Phase Spectrum ----------
print("\n --- [A9]: Plotting Phase Spectrum --- ")

plt.figure(figsize=(10, 5))
plt.plot(freqs/1e6, phase, linewidth=1)
plt.title("A9: Phase Spectrum (Unwrapped)")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Phase (rad)")
plt.grid(True)

# mark the peak
plt.axvline(peak_hz/1e6, color='red', linestyle='--', label=f"Peak ~ {peak_hz/1e6:.6f} MHz")
plt.legend()

plt.tight_layout()
plt.savefig("a9_phase.png", dpi=150)
print("Saved: a9_phase.png")

# ---------- STEP A9-IMPEDANCE: Approximate Impedance Plot ----------
print("\n --- [A9]: Plotting Approximate Impedance View ---")

eps = 1e-12  # numerical guard
Z = 1.0 / (mag + eps)

plt.figure(figsize=(10, 5))
plt.plot(freqs/1e6, Z, linewidth=1)
plt.title("A9: Impedance Approximation (1 / Magnitude)")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Impedance (1/mag)")
plt.grid(True)

# Highlight the peak region
plt.axvline(peak_hz/1e6, color='red', linestyle='--',
            label=f"Peak ~ {peak_hz/1e6:.6f} MHz")
plt.legend()

plt.tight_layout()
plt.savefig("a9_impedance.png", dpi=150)
print("Saved: a9_impedance.png")

# ---------- STEP A9-COMBINED PLOT: Mag + Phase + Impedance ----------
print("\n --- [A9]: Creating Combined Spectrum Figure ---")

fig, axes = plt.subplots(3, 1, figsize=(10, 12))

# --- Magnitude ---
axes[0].plot(freqs/1e6, mag, linewidth=1)
axes[0].set_title("Magnitude Spectrum")
axes[0].set_xlabel("Frequency (MHz)")
axes[0].set_ylabel("Magnitude")
axes[0].grid(True)
axes[0].axvline(peak_hz/1e6, color='red', linestyle='--')

# --- Phase ---
axes[1].plot(freqs/1e6, phase, linewidth=1)
axes[1].set_title("Phase Spectrum (Unwrapped)")
axes[1].set_xlabel("Frequency (MHz)")
axes[1].set_ylabel("Phase (rad)")
axes[1].grid(True)
axes[1].axvline(peak_hz/1e6, color='red', linestyle='--')

# --- Impedance ---
axes[2].plot(freqs/1e6, Z, linewidth=1)
axes[2].set_title("Impedance Approximation (1/mag)")
axes[2].set_xlabel("Frequency (MHz)")
axes[2].set_ylabel("Impedance (arb. units)")
axes[2].grid(True)
axes[2].axvline(peak_hz/1e6, color='red', linestyle='--')

plt.tight_layout()
plt.savefig("a9_combined.png", dpi=150)
print("Saved: a9_combined.png")
print("A9 completed.\n")




