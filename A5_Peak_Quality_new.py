# ------------------------------------------------------------
# A5 – PEAK QUALITY & SNR EVALUATION
#
# Purpose:
#   This file evaluates the MLA frame quality by analysing its
#   FFT spectrum. It finds the resonance peak, estimates the
#   noise floor from a region away from the peak, and computes
#   the Signal-to-Noise Ratio (SNR). This helps determine whether
#   the MLA signal is reliable for overtone tracking.
#
# What this file does:
#   - Loads MLA frame (raw or simulated)
#   - Computes FFT magnitude using Hann window
#   - Detects resonance peak
#   - Computes signal power and noise power
#   - Computes SNR using MLA equations
#   - Categorizes signal as FAIL / POOR / GOOD / EXCELLENT
#   - Prints warnings AND fixes with references to documentation
#   - Saves CSV & a5_spectrum.png plot
#
# Equations used in MLA Peak Quality:
#   |X[k]|     = |FFT(x[n] * window[n])|
#   P_signal   = mean( |X[k]|^2 around peak )
#   P_noise    = mean( |X[k]|^2 in noise region )
#   SNR_linear = P_signal / P_noise
#   SNR_dB     = 10 * log10( SNR_linear )
#
# These equations are printed once here for memory.
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import csv
import json
import sys
import os

print("\n--------------------------------------------------")
print("A5 – Peak Quality & SNR Evaluation")
print("--------------------------------------------------\n")

print("Equations used:")
print("|X[k]| = |FFT(x[n] * window[n])|")
print("P_signal = mean(|X[k]|^2 around peak)")
print("P_noise  = mean(|X[k]|^2 in noise region)")
print("SNR_linear = P_signal / P_noise")
print("SNR_dB = 10 * log10(SNR_linear)\n")

# ------------------------------------------------------------
# MODE SELECTION (same MLA structure)
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
    fs = 125000000
    t = np.linspace(0, 0.0001, 40000)
    signal = 0.01*np.sin(2*np.pi*15e6*t) + 0.001*np.random.randn(len(t))

elif OLD_PYTHON_FILE_MODE:
    if not os.path.exists(RAW_FILE):
        print("ERROR: Raw file not found.")
        sys.exit(1)
    data = np.fromfile(RAW_FILE, dtype=np.int16)
    signal = data.astype(float)
    fs = 125000000

elif NEW_PYTHON_FILE_MODE:
    if not os.path.exists(RAW_FILE) or not os.path.exists(JSON_FILE):
        print("ERROR: Missing raw/json file.")
        sys.exit(1)

    data = np.fromfile(RAW_FILE, dtype=np.int16)
    signal = data.astype(float)

    with open(JSON_FILE, "r") as f:
        meta = json.load(f)
    fs = meta.get("sample_rate_hz", 125000000)

else:
    print("ERROR: Select a mode.")
    sys.exit(1)

# ------------------------------------------------------------
# PROCESSING BEGINS
# ------------------------------------------------------------

# Remove DC offset
signal = signal - np.mean(signal)

# Frame length
N = len(signal)

# Apply Hann window
window = np.hanning(N)
signal_w = signal * window

# FFT magnitude with correct Hann normalization
X = np.fft.rfft(signal_w)
mag = np.abs(X) / (np.sum(window) / 2)
mag_db = 20 * np.log10(mag + 1e-12)

# Frequency axis
freqs = np.fft.rfftfreq(N, 1/fs)

# ------------------------------------------------------------
# PEAK DETECTION
# ------------------------------------------------------------

peak_bin = int(np.argmax(mag))
peak_freq_hz = freqs[peak_bin]

# Signal power around peak (7 bins)
lo = max(peak_bin - 3, 0)
hi = min(peak_bin + 4, len(mag))
P_signal = np.mean(mag[lo:hi] ** 2)

# ------------------------------------------------------------
# NOISE ESTIMATION
# ------------------------------------------------------------

# Noise region far from resonance


bin_hz = fs / N

# skip 1 MHz around the peak
skip_bins = int(1e6 / bin_hz)

# left side noise region (far from peak)
left_start = max(0, peak_bin - 20*skip_bins)
left_end   = max(0, peak_bin - skip_bins)
left_noise = mag[left_start:left_end]

# right side noise region (far from peak)
right_start = min(len(mag), peak_bin + skip_bins)
right_end   = min(len(mag), peak_bin + 20*skip_bins)
right_noise = mag[right_start:right_end]

# combine both sides
if len(left_noise) + len(right_noise) == 0:
    noise_mag = mag[10:50]     # fallback (never happens normally)
else:
    noise_mag = np.concatenate([left_noise, right_noise])

P_noise = np.mean(noise_mag ** 2)
noise_floor_db = 10 * np.log10(P_noise + 1e-12)


# ------------------------------------------------------------
# SNR COMPUTATION
# ------------------------------------------------------------

SNR_linear = P_signal / P_noise
SNR_dB = 10 * np.log10(SNR_linear + 1e-12)

# ------------------------------------------------------------
# QUALITY LABEL
# ------------------------------------------------------------

if SNR_dB < 5:
    quality = "FAIL"
elif SNR_dB < 10:
    quality = "POOR"
elif SNR_dB < 15:
    quality = "GOOD"
else:
    quality = "EXCELLENT"

# ------------------------------------------------------------
# WARNINGS + FIXES (your requirement)
# ------------------------------------------------------------

warnings_list = []
fix_list = []

# Low SNR
if SNR_dB < 5:
    warnings_list.append("Low SNR – MLA signal is weak or noisy.")
    fix_list.append("Fix: Increase DAC amplitude → edit AMP in test.py (signal generation). Refer to MLA Documentation (Section: DAC Amplitude).")

# Unrealistic high SNR
if SNR_dB > 25:
    warnings_list.append("SNR extremely high – unrealistic for MLA hardware.")
    fix_list.append("Fix: Check FFT scaling or metadata sample rate. Refer to MLA Documentation (Section: Metadata & FFT).")

# Noise floor too low
if noise_floor_db < -110:
    warnings_list.append("Noise floor too low – not physically realistic.")
    fix_list.append("Fix: Verify ADC scaling and Hann window normalization. Refer to MLA Documentation (Signal Scaling section).")

# ------------------------------------------------------------
# PRINT RESULTS
# ------------------------------------------------------------

print(f"Peak Frequency : {peak_freq_hz/1e6:.6f} MHz")
print(f"SNR            : {SNR_dB:.2f} dB")
print(f"Quality        : {quality}")

for w in warnings_list:
    print(w)
for fx in fix_list:
    print(fx)

# ------------------------------------------------------------
# SAVE CSV
# ------------------------------------------------------------

with open("a5_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["peak_freq_hz", "snr_db", "quality", "noise_floor_db"])
    writer.writerow([peak_freq_hz, SNR_dB, quality, noise_floor_db])

print("Saved: a5_results.csv")

# ------------------------------------------------------------
# SAVE PLOT
# ------------------------------------------------------------

plt.figure(figsize=(10,5))
plt.plot(freqs/1e6, mag_db, label="FFT Magnitude (dB)")
plt.axvline(peak_freq_hz/1e6, color='r', linestyle='--', label="Peak")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Magnitude (dB)")
plt.title(f"A5 Spectrum (SNR = {SNR_dB:.1f} dB)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("a5_spectrum.png", dpi=200)
plt.show()

print("Saved: a5_spectrum.png")

# ------------------------------------------------------------
# COMPLETION
# ------------------------------------------------------------

print("\nA5 completed.\n")

