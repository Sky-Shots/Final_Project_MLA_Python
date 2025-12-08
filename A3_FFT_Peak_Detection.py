# --------------------------------------------------------------------
# A3 – FFT Peak Detection for MLA Signal
#
# Description:
# This file performs the first frequency-domain analysis step in the
# MLA pipeline. After collecting the time-domain waveform from the
# FPGA (either simulated or real MLA output), A3 computes the FFT
# and identifies the strongest frequency component. This peak is the
# estimated resonance frequency of the QCM sensor for the selected
# overtone.
#
# The purpose of A3 is to verify that a clear and dominant resonance
# peak exists before attempting more advanced MLA processing such as
# smoothing, quality estimation, reliability checks, or overtone
# tracking. A well-defined peak indicates that the excitation signal
# and acquisition chain are functioning correctly.
#
# A3 carries out the following:
#   • Reads MLA data (either simulation, old test.py format, or
#     new test_new.py format)
#   • Removes DC offset from the signal
#   • Applies a Hann window to reduce spectral leakage
#   • Computes the one-sided FFT
#   • Extracts the frequency with maximum magnitude
#   • Saves the top N strongest peaks to CSV
#   • Plots the full FFT and a zoomed window around the peak
#
# Equations used:
#
#   1. DC removal:
#        x_clean[n] = x[n] − mean(x)
#
#   2. Windowing (Hann window):
#        w[n] = 0.5 * (1 − cos(2πn / (N−1)))
#        x_win[n] = x_clean[n] * w[n]
#
#   3. FFT:
#        X[k] = Σ x_win[n] * exp(−j 2π k n / N)
#
#   4. Magnitude spectrum:
#        |X[k]| = sqrt( Re(X[k])² + Im(X[k])² )
#
#   5. Peak frequency:
#        k_peak = argmax( |X[k]| )
#        f_peak = k_peak * (fs / N)
#
# These equations correspond to the initial spectral estimation step
# used in MLA. A sharp, clean peak in A3 indicates that the QCM is
# oscillating properly and that the later MLA steps (A4–A9) will be
# meaningful.
#
# --------------------------------------------------------------------



import numpy as np
import json
import os
import sys
import csv
import matplotlib.pyplot as plt

print("\n--------------------------------------------------")
print("A3 – FFT Peak Detection for MLA Signal")
print("--------------------------------------------------\n")

# === FILE PATHS ===
RAW_FILE  = "mla_data.raw"
META_FILE = "mla_data.json"

# === MODE SWITCHES ===
SIM_MODE             = True      # Synthetic test signal
NEW_PYTHON_FILE_MODE = False     # test_new.py format (RAW + JSON)
OLD_PYTHON_FILE_MODE = False     # test.py format (RAW only)

# ------------------------------------------------------------
# MODE: SIMULATION
# ------------------------------------------------------------
if SIM_MODE:
    # SIM: self-contained test data
    fs = 125_000_000                  # sampling rate
    N  = 16384                        # samples
    t  = np.arange(N) / fs
    f0 = 15_000_000                   # test tone
    x  = np.sin(2 * np.pi * f0 * t).astype(np.float64)

else:
    # --------------------------------------------------------
    # MODE: REAL MLA FILES
    # --------------------------------------------------------
    if NEW_PYTHON_FILE_MODE:
        print("NEW_PYTHON_FILE_MODE = TRUE --> loading new MLA format")

        # check files exist
        if not os.path.exists(RAW_FILE) or not os.path.exists(META_FILE):
            print(f"ERROR: Missing {RAW_FILE} or {META_FILE}")
            sys.exit(1)

        # load JSON metadata
        with open(META_FILE, 'r') as f:
            meta = json.load(f)

        fs      = float(meta["sample_rate_hz"])
        snapped = np.array(meta.get("snapped_freqs_hz", []), dtype=float)

        # load int16 raw
        def load_int16(fname, max_samples=None):
            with open(fname, "rb") as f:
                data = np.frombuffer(f.read(), dtype=np.int16)
            if max_samples is not None:
                data = data[:max_samples]
            return data

        x = load_int16(RAW_FILE)
        if x.size < 1024:
            print("ERROR: not enough samples")
            sys.exit(1)

    elif OLD_PYTHON_FILE_MODE:
        print("OLD_PYTHON_FILE_MODE = TRUE --> loading old MLA format")
        meta = None  # old test.py has no metadata

        with open(RAW_FILE, 'rb') as f:
            x = np.frombuffer(f.read(), dtype=np.int16)

        fs = 125_000_000  # fallback sample rate

        if x.size < 1024:
            print("ERROR: not enough samples")
            sys.exit(1)

    else:
        raise ValueError("Enable either NEW_PYTHON_FILE_MODE or OLD_PYTHON_FILE_MODE")

# ------------------------------------------------------------
# Display sampling info
# ------------------------------------------------------------
if SIM_MODE:
    print(f"Sampling Rate : {fs} Hz; Test Tone : {f0} Hz; Nyquist : {fs/2} Hz")
else:
    print(f"Sampling Rate : {fs} Hz; Nyquist : {fs/2} Hz")

# ------------------------------------------------------------
# STEP 4: Detrend and window
# ------------------------------------------------------------
x = x.astype(np.float64)
x = x - np.mean(x)            # remove DC
w = np.hanning(x.size)        # Hann window

# ------------------------------------------------------------
# STEP 5: FFT + Peak
# ------------------------------------------------------------
X       = np.fft.rfft(x * w)
freqs   = np.fft.rfftfreq(x.size, d=1.0 / fs)
mag     = np.abs(X) / (x.size / 2)

k       = np.argmax(mag)
peak_hz = freqs[k]
print("Peak Frequency (Hz):", float(peak_hz))

# ------------------------------------------------------------
# STEP 6A: Save top-N peaks to CSV
# ------------------------------------------------------------
TOP_N   = 5
CSV_OUT = "a3_peaks.csv"

idx_sort = np.argsort(mag)[::-1]
start_i = 1 if idx_sort[0] == 0 else 0

print("\n rank |     freq (Hz)        |     mag")
print("-------+----------------------+---------")
for r, k in enumerate(idx_sort[start_i:start_i+TOP_N], start=1):
    print(f"{r:>5d}  | {freqs[k]:>22.1f} | {mag[k]:>7.3f}")

with open(CSV_OUT, 'w', newline="") as f:
    wcsv = csv.writer(f)
    wcsv.writerow(["rank", "freq_hz", "magnitude"])
    for r, k in enumerate(idx_sort[start_i:start_i+TOP_N], start=1):
        wcsv.writerow([r, freqs[k], mag[k]])

print(f"Saved CSV: {CSV_OUT}")

# ------------------------------------------------------------
# STEP 6B: Plotting
# ------------------------------------------------------------

# Full spectrum
plt.figure()
plt.plot(freqs / 1e6, mag)
plt.xlabel("Frequency (MHz)")
plt.ylabel("Magnitude")
plt.title("A3: Spectrum (one-sided)")
plt.tight_layout()
plt.savefig("a3_spectrum_full.png", dpi=150)
print("Saved: a3_spectrum_full.png")

# Zoom ±3 MHz around peak
span = 3e6
plt.figure()
plt.plot(freqs / 1e6, mag)
plt.xlim((peak_hz - span) / 1e6, (peak_hz + span) / 1e6)
plt.xlabel("Frequency (MHz)")
plt.ylabel("Magnitude")
plt.title(f"A3: Zoom near peak ~ {peak_hz/1e6:.3f} MHz")
plt.tight_layout()
plt.savefig("a3_spectrum_zoom.png", dpi=150)
print("Saved: a3_spectrum_zoom.png")

plt.show()
