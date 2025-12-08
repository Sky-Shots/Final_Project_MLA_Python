
# ------------------------------------------------------------
# A7 – AMPLITUDE & CLIPPING SAFETY ANALYSIS (MLA Pipeline)
#
# Purpose:
#   This file evaluates whether the MLA time-domain waveform
#   is operating within a safe amplitude range and whether
#   the signal is suffering from clipping in the ADC.
#
# Why amplitude & clipping safety matter:
#   The MLA method relies on an undistorted sinusoidal drive.
#   If the ADC saturates (clipping), the FFT becomes distorted,
#   frequency estimation becomes inaccurate, and all steps
#   downstream (A5 SNR, A6 reliability, A8 comb-size, A9 phase)
#   will produce unreliable results.
#
#   Therefore, A7 checks:
#       • Is the waveform clipped?
#       • Is the amplitude too high?
#       • Is the amplitude too low?
#
# What this file evaluates:
#   A) Peak amplitude after normalization:
#        peak = max(|x[n]|)
#
#   B) Clipping:
#        clipping = (peak >= 1.0)
#
#   C) Safe amplitude region:
#        SAFE if 0.05 ≤ peak ≤ 0.90
#        TOO LOW if peak < 0.05  (signal too weak)
#        TOO HIGH if peak > 0.90 (risk of clipping)
#
#   D) SIM MODE (debugging only):
#        Sweeps 2 amplitudes: AMP = 1.0 and AMP = 0.9
#        ONLY for testing behaviour, NOT real MLA usage.
#
# Important:
#   Amplitude SHOULD NOT be changed in A7.
#   The true DAC amplitude is controlled in:
#       → test.py / test_new.py (AMP value)
#   A7 only ANALYSES the resulting waveform.
#
# Relevant equations (printed once below):
#
#   (E1) Normalization:
#         x_norm[n] = x_raw[n] / max(|x_raw|)
#
#   (E2) Peak amplitude:
#         peak = max(|x_norm[n]|)
#
#   (E3) Clipping condition:
#         clipping = (peak >= 1.0)
#
#   (E4) DAC→ADC mapping:
#         ADC_peak ≈ DAC_amp * Transfer_Gain
#
#   (E5) Safe region rule:
#         safe if 0.05 ≤ peak ≤ 0.90
#
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import csv
import json
import os
import sys

print("\n--------------------------------------------------")
print("A7 – Amplitude & Clipping Safety Test")
print("--------------------------------------------------\n")

# ------------------------------------------------------------
# Print equations once
# ------------------------------------------------------------
print("Equations used in A7:")
print("x_norm[n] = x_raw[n] / max(|x_raw|)")
print("peak      = max(|x_norm[n]|)")
print("clipping  = (peak >= 1.0)")
print("Safe amplitude region: 0.05 ≤ peak ≤ 0.90\n")

# ------------------------------------------------------------
# Mode selection
# ------------------------------------------------------------
SIM_MODE = False
OLD_PYTHON_FILE_MODE = True
NEW_PYTHON_FILE_MODE = False

RAW_FILE = "mla_data.raw"
JSON_FILE = "mla_data.json"

# ------------------------------------------------------------
# LOAD SIGNAL
# ------------------------------------------------------------

if SIM_MODE:
    print("SIM MODE active: generating artificial test signals...")
    fs = 125000000
    t = np.linspace(0, 0.00004, 40000)

    # We will sweep two amplitudes for demonstration
    sim_amplitudes = [1.0, 0.9]
    results = []

    for amp in sim_amplitudes:
        sim_signal = amp * np.sin(2*np.pi*15e6*t)
        results.append((amp, sim_signal))

else:
    # Real MLA data mode
    if OLD_PYTHON_FILE_MODE:
        if not os.path.exists(RAW_FILE):
            print("ERROR: Raw MLA file missing.")
            sys.exit(1)
        data = np.fromfile(RAW_FILE, dtype=np.int16)
        signal = data.astype(float)
        fs = 125000000

    elif NEW_PYTHON_FILE_MODE:
        if not os.path.exists(RAW_FILE) or not os.path.exists(JSON_FILE):
            print("ERROR: Missing raw/json files.")
            sys.exit(1)

        data = np.fromfile(RAW_FILE, dtype=np.int16)
        signal = data.astype(float)

        with open(JSON_FILE, "r") as f:
            meta = json.load(f)

        fs = meta.get("sample_rate_hz", 125000000)

    else:
        print("ERROR: No MLA mode selected.")
        sys.exit(1)

# ------------------------------------------------------------
# REAL MODE PROCESSING
# ------------------------------------------------------------
if not SIM_MODE:

    # Remove DC offset
    signal = signal - np.mean(signal)

    # Normalize
    max_val = np.max(np.abs(signal))
    signal_norm = signal / max_val if max_val != 0 else signal

    # Peak amplitude
    peak_val = np.max(np.abs(signal_norm))

    # Clipping test
    clipping = (peak_val >= 1.0)

    # Determine safety category
    if clipping:
        status = "CLIPPED"
    elif peak_val < 0.05:
        status = "TOO LOW"
    elif peak_val > 0.90:
        status = "TOO HIGH"
    else:
        status = "SAFE"

    print(f"Peak amplitude : {peak_val:.4f}")
    print(f"Clipping       : {clipping}")
    print(f"Amplitude zone : {status}")

    # Warnings + Fixes
    if clipping:
        print("Warning: ADC CLIPPING – waveform saturated.")
        print("Fix: Reduce DAC amplitude (AMP in test.py).")
        print("Refer to MLA Documentation → DAC Amplitude section.\n")

    if status == "TOO LOW":
        print("Warning: Signal amplitude is too weak.")
        print("Fix: Increase AMP in test.py.")
        print("Refer to MLA Documentation → Signal Strength.\n")

    if status == "TOO HIGH" and not clipping:
        print("Warning: Amplitude close to clipping – unstable region.")
        print("Fix: Reduce AMP slightly in test.py.")
        print("Refer to MLA Documentation → DAC Safety section.\n")

    # SAVE CSV
    with open("a7_amp_safety.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["peak_amplitude", "clipping", "status"])
        writer.writerow([peak_val, clipping, status])

    print("Saved: a7_amp_safety.csv")

    # PLOT
    plt.figure(figsize=(6,4))
    plt.bar(["peak amplitude"], [peak_val], color="purple")
    plt.axhline(1.0, color="red", linestyle="--", label="Clipping level (1.0)")
    plt.axhline(0.90, color="orange", linestyle="--", label="Upper safe limit 0.90")
    plt.axhline(0.05, color="green", linestyle="--", label="Lower safe limit 0.05")
    plt.title("A7 – Amplitude & Clipping Safety")
    plt.ylabel("Normalized Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig("a7_amp_safety.png", dpi=200)
    plt.show()

    print("Saved: a7_amp_safety.png\n")
    print("A7 completed.\n")

# ------------------------------------------------------------
# SIM MODE PROCESSING (debugging)
# ------------------------------------------------------------
else:
    sim_peaks = []

    for amp, sig in results:

        # DC removal
        sig = sig - np.mean(sig)

        # Normalize
        m = np.max(np.abs(sig))
        sig_norm = sig / m if m!=0 else sig

        peak_val = np.max(np.abs(sig_norm))
        clipping = (peak_val >= 1.0)

        sim_peaks.append((amp, peak_val, clipping))

        print(f"Test amplitude: {amp:.2f}")
        print(f"Normalized peak: {peak_val:.4f}")
        print(f"Clipping       : {clipping}\n")

    # Plot SIM MODE
    amps = [x[0] for x in sim_peaks]
    peaks = [x[1] for x in sim_peaks]

    plt.figure(figsize=(6,4))
    plt.plot(amps, peaks, marker='o')
    plt.axhline(1.0, color="red", linestyle="--", label="Clipping level (1.0)")
    plt.axhline(0.90, color="orange", linestyle="--", label="Upper safe limit 0.90")
    plt.axhline(0.05, color="green", linestyle="--", label="Lower safe limit 0.05")
    plt.xlabel("Simulated DAC Amplitude (relative)")
    plt.ylabel("Peak ADC Amplitude")
    plt.title("A7 – SIM MODE Amplitude Response")
    plt.legend()
    plt.tight_layout()
    plt.savefig("a7_amp_safety_sim.png", dpi=200)
    plt.show()

    print("Saved: a7_amp_safety_sim.png\n")
    print("A7 completed.\n")

