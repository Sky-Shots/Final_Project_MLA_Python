#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ------------------------------------------------------------
# A8 – BANDWIDTH & COMB-SIZE EVALUATION (MLA Pipeline)
#
# Purpose:
#   A8 evaluates how well the MLA tones (overtone frequencies)
#   are positioned inside the FFT bins, and whether the bandwidth
#   of the measurement is appropriate for the selected comb-size.
#   This step checks if the MLA tone(s) land close to their
#   expected snapped frequencies and whether the SNR around each
#   tone is sufficient.
#
# Why A8 matters in MLA:
#   The MLA method can evaluate multiple tones (overtone numbers)
#   in the same acquisition frame. Each tone must land correctly
#   in the FFT spectrum. When metadata fails to store the snapped
#   frequencies or when only one tone is used, A8 detects this.
#
# What this file evaluates:
#   (E1) FFT bin frequency:
#        f_bin[k] = k * (fs / N)
#
#   (E2) Tone alignment error:
#        error = |f_target - f_bin_peak|
#
#   (E3) Local SNR:
#        SNR_local = peak_dB - noise_dB
#
#   (E4) Only one tone:
#        → comb-size plot collapses to a single dot.
#
#   (E5) Missing metadata:
#        → fallback frequency = 15e6 Hz
#
# Relevant thesis equations (printed for reference):
#
#   Kanazawa–Gordon (liquid loading):
#       Δf = -( f0^(3/2) / sqrt(π μq ρq) ) * sqrt(ηL ρL)
#
#   Sauerbrey (mass loading):
#       Δf = -(2 f0^2 / (A sqrt(ρq μq))) * Δm
#
# These relations explain bandwidth broadening and shifts.
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import csv
import json
import sys
import os

print("\n--------------------------------------------------")
print("A8 – Bandwidth & Comb-Size Evaluation")
print("--------------------------------------------------\n")

# ------------------------------------------------------------
# Print equations ONCE
# ------------------------------------------------------------
print("Equations used in A8:")
print("f_bin[k]    = k * (fs / N)")
print("error       = |f_target - f_bin_peak|")
print("SNR_local   = peak_dB - noise_dB")
print("Kanazawa–Gordon: Δf = -(f0^(3/2)/sqrt(π μq ρq)) * sqrt(ηL ρL)")
print("Sauerbrey:       Δf = -(2 f0^2 / (A sqrt(ρq μq))) * Δm\n")

# ------------------------------------------------------------
# MODE SELECTION
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
    # Artificial debug data: multiple tones
    print("SIM MODE active.")
    fs = 125000000
    t = np.linspace(0, 0.00004, 50000)
    sig = (
        0.8*np.sin(2*np.pi*15e6*t) +
        0.5*np.sin(2*np.pi*16e6*t) +
        0.3*np.sin(2*np.pi*17e6*t)
    )
    signal = sig
    snapped_freqs = [15e6, 16e6, 17e6]

elif OLD_PYTHON_FILE_MODE:
    if not os.path.exists(RAW_FILE):
        print("ERROR: Raw file missing.")
        sys.exit(1)

    data = np.fromfile(RAW_FILE, dtype=np.int16)
    signal = data.astype(float)
    fs = 125000000

    # OLD MODE has NO JSON → no snapped frequencies
    snapped_freqs = None

elif NEW_PYTHON_FILE_MODE:
    if not os.path.exists(RAW_FILE) or not os.path.exists(JSON_FILE):
        print("ERROR: RAW/JSON missing.")
        sys.exit(1)

    data = np.fromfile(RAW_FILE, dtype=np.int16)
    signal = data.astype(float)

    with open(JSON_FILE, "r") as f:
        meta = json.load(f)

    fs = meta.get("sample_rate_hz", 125000000)
    snapped_freqs = meta.get("snapped_freqs_hz", None)

else:
    print("ERROR: No mode selected.")
    sys.exit(1)

# ------------------------------------------------------------
# HANDLE MISSING METADATA
# ------------------------------------------------------------
if snapped_freqs is None or len(snapped_freqs) == 0:
    print("[WARN] snapped_freqs_hz missing → using fallback [15e6]")
    print("Fix: Ensure metadata is updated in test scripts")
    print("Refer to MLA Documentation → Metadata section.\n")
    snapped_freqs = [15e6]  # fallback frequency

# ------------------------------------------------------------
# PRE-PROCESSING (normalize + FFT)
# ------------------------------------------------------------
signal = signal - np.mean(signal)
N = len(signal)

window = np.hanning(N)
sig_w = signal * window

X = np.fft.rfft(sig_w)
mag = np.abs(X) / (np.sum(window)/2)
mag_dB = 20*np.log10(mag + 1e-12)
freqs = np.fft.rfftfreq(N, 1/fs)

bin_width = fs / N

# ------------------------------------------------------------
# EVALUATE EACH TONE
# ------------------------------------------------------------
tone_errors = []
tone_snrs = []

for f_target in snapped_freqs:

    # Find closest FFT bin to target frequency
    idx = np.argmin(np.abs(freqs - f_target))
    
    
        # sanity guard if idx invalid
    if idx < 0 or idx >= len(mag_dB):
        print(f"[WARN] Invalid bin for target {f_target} Hz")
        tone_errors.append(np.nan)
        tone_snrs.append(np.nan)
        continue
        
    peak_val = mag_dB[idx]

    # Noise region for SNR (simple static region)
    if len(mag_dB) > 800:
        noise_level = np.mean(mag_dB[200:800])
    else:
        noise_level = np.mean(mag_dB[10:50])

    snr_local = peak_val - noise_level

    bin_error = abs(freqs[idx] - f_target)

    tone_errors.append(bin_error)
    tone_snrs.append(snr_local)

# ------------------------------------------------------------
# PRINT RESULTS
# ------------------------------------------------------------
print("Detected tones:", snapped_freqs)
print("Bin width:", bin_width)

for i, f_target in enumerate(snapped_freqs):
    print(f"\nTone {i+1}: {f_target/1e6:.3f} MHz")
    print(f"  Alignment error : {tone_errors[i]:.2f} Hz")
    print(f"  Local SNR       : {tone_snrs[i]:.2f} dB")

    # WARNINGS & FIXES
    if abs(tone_errors[i]) > bin_width:
        print("  Warning: tone misaligned relative to FFT grid.")
        print("  Fix: Increase FFT length or adjust desired bandwidth.")
        print("  Refer: Documentation → Comb-size & Bandwidth.\n")

    if tone_snrs[i] < 5:
        print("  Warning: Low SNR for this tone.")
        print("  Fix: Increase DAC amplitude (AMP in test.py).")
        print("  Refer: Documentation → Signal Strength.\n")
        
# ------------------------------------------------------------
# COMB-SIZE PLOT (single tone → single dot)
# ------------------------------------------------------------
plt.figure(figsize=(6,5))
plt.scatter(snapped_freqs, tone_snrs, c="blue", s=80)

plt.xlabel("Tone Frequency (Hz)")
plt.ylabel("Local SNR (dB)")
plt.title("A8 – Comb-SNR Plot")

plt.grid()
plt.tight_layout()
plt.savefig("a8_comb_snr.png", dpi=200)
plt.show()

print("Saved: a8_comb_snr.png\n")

# ------------------------------------------------------------
# SAVE CSV
# ------------------------------------------------------------
with open("a8_comb_check.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["f_target_hz", "bin_error_hz", "snr_db"])
    for ft, err, snr in zip(snapped_freqs, tone_errors, tone_snrs):
        writer.writerow([ft, err, snr])

print("Saved: a8_comb_check.csv\n")
print("A8 completed.\n")

