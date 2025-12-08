
# ------------------------------------------------------------
# A6 – RELIABILITY TEST (Time-Domain Stability of MLA Signal)
#
# Purpose:
#   The MLA system drives the QCM sensor with a sinusoidal signal
#   and measures the resulting oscillation through the ADC. A
#   reliable MLA frame must show a smooth, stable, continuous
#   waveform. Sudden jumps, clipping, jitter, or inconsistent
#   transitions indicate instability either in the QCM oscillation,
#   electrical chain, or experimental setup.
#
# Why reliability matters:
#   The MLA method depends on extracting resonance frequency and
#   dissipation from the FFT of the time-domain signal. If the
#   waveform is unstable, the FFT peak becomes distorted, causing
#   errors in A3 (peak detection), A4 (tracking), A5 (SNR), and
#   A7–A9. Therefore, reliability is a prerequisite before any
#   frequency-domain analysis.
#
# What this file evaluates:
#   1. ADC Clipping:
#        x[n] >= 1.0  or  x[n] <= -1.0   (after normalization)
#        This indicates the ADC has reached its limit.
#
#   2. Step-size smoothness:
#        step[n] = | x[n] - x[n-1] |
#        max_step = max(step[n])
#
#        Reliability threshold:
#            Reliable   if max_step < 0.015
#            Unreliable if max_step >= 0.015
#
#        Threshold chosen from MLA hardware behaviour, TD0 zoom
#        observations, and real data collected from your board.
#
# Theoretical background (Thesis context):
#
#   • Kanazawa–Gordon relation (liquid loading instability):
#       Δf = -( f0^(3/2) / sqrt(π μq ρq) ) * sqrt(ηL ρL)
#     Instabilities in viscosity or density cause abrupt QCM
#     oscillation changes, visible as step jumps.
#
#   • Sauerbrey equation (mass loading instability):
#       Δf = -( 2 f0^2 / (A sqrt(ρq μq)) ) * Δm
#     Unstable mass attachment or vibration noise produces jitter
#     detectable in the time-domain.
#
# Connection to MLA pipeline:
#   A6 validates the signal BEFORE frequency extraction. If A6 fails,
#   results from A7 (amplitude safety), A8 (comb-size), and A9 (phase
#   profile) may become misleading. A6 ensures the foundation of the
#   MLA frame is correct before proceeding.
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import csv
import json
import sys
import os

print("\n--------------------------------------------------")
print("A6 – Reliability Test (MLA Signal Stability)")
print("--------------------------------------------------\n")

# ------------------------------------------------------------
# Print equations ONCE (top of file)
# ------------------------------------------------------------
print("Equations used:")
print("step[n]     = |x[n] - x[n-1]|")
print("max_step    = max(step[n])")
print("Reliable if max_step < 0.015")
print("Kanazawa–Gordon: Δf = -(f0^(3/2) / sqrt(π μq ρq)) * sqrt(ηL ρL)")
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
    fs = 125000000
    t = np.linspace(0, 0.0001, 40000)
    signal = 0.2*np.sin(2*np.pi*15e6*t) + 0.01*np.random.randn(len(t))

elif OLD_PYTHON_FILE_MODE:
    if not os.path.exists(RAW_FILE):
        print("ERROR: Raw file missing.")
        sys.exit(1)
    data = np.fromfile(RAW_FILE, dtype=np.int16)
    signal = data.astype(float)
    fs = 125000000

elif NEW_PYTHON_FILE_MODE:
    if not os.path.exists(RAW_FILE) or not os.path.exists(JSON_FILE):
        print("ERROR: Missing RAW or JSON.")
        sys.exit(1)
    data = np.fromfile(RAW_FILE, dtype=np.int16)
    signal = data.astype(float)
    with open(JSON_FILE, "r") as f:
        meta = json.load(f)
    fs = meta.get("sample_rate_hz", 125000000)

else:
    print("ERROR: No mode selected.")
    sys.exit(1)

# ------------------------------------------------------------
# NORMALIZE SIGNAL
# ------------------------------------------------------------
signal = signal - np.mean(signal)
peak = np.max(np.abs(signal))
signal_norm = signal / peak if peak != 0 else signal

# ------------------------------------------------------------
# FUNCTION 1 — CLIPPING
# ------------------------------------------------------------
def check_clipping(x):
    """Returns True if waveform reaches ADC limits (±1.0)."""
    return np.any(x >= 1.0) or np.any(x <= -1.0)

# ------------------------------------------------------------
# FUNCTION 2 — STEP SIZE (smoothness)
# ------------------------------------------------------------
def compute_max_step(x):
    """Computes max |x[n] - x[n-1]| to detect sudden jumps."""
    steps = np.abs(np.diff(x))
    return np.max(steps), steps

# ------------------------------------------------------------
# RUN RELIABILITY TEST
# ------------------------------------------------------------
clip_flag = check_clipping(signal_norm)
max_step, steps = compute_max_step(signal_norm)
THR = 0.015
reliable = (max_step < THR)

# ------------------------------------------------------------
# PRINT RESULTS
# ------------------------------------------------------------
print(f"Max step size : {max_step:.5f}")
print(f"Clipping      : {clip_flag}")
print(f"Reliability   : {'GOOD' if reliable else 'FAIL'}")

# ------------------------------------------------------------
# WARNINGS + FIXES
# ------------------------------------------------------------
warnings = []
fixes = []

if clip_flag:
    warnings.append("Warning: Clipping detected – ADC saturation.")
    fixes.append("Fix: Reduce DAC amplitude (AMP in test.py). Refer to MLA Documentation → DAC Amplitude.")

if max_step >= THR:
    warnings.append("Warning: Sudden jumps detected – signal instability.")
    fixes.append("Fix: Check grounding, cables, sensor contact. Reduce external noise. Refer to MLA Documentation → Reliability section.")

if not warnings:
    print("\nAll reliability checks passed.\n")
else:
    for w in warnings:
        print(w)
    for f in fixes:
        print(f)

# ------------------------------------------------------------
# SAVE CSV
# ------------------------------------------------------------
with open("a6_reliability.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["max_step", "clipping", "reliable"])
    writer.writerow([max_step, clip_flag, reliable])

print("Saved: a6_reliability.csv")

# ------------------------------------------------------------
# SAVE PLOT
# ------------------------------------------------------------
plt.figure(figsize=(6,4))
plt.bar(["max_step"], [max_step], color="blue")
plt.axhline(THR, color="red", linestyle="--", label=f"Threshold ({THR})")
plt.title("A6 Reliability – Max Step Size")
plt.ylabel("Step size")
plt.legend()
plt.tight_layout()
plt.savefig("a6_maxstep_bar.png", dpi=200)
plt.show()

print("Saved: a6_maxstep_bar.png\n")
print("A6 completed.\n")

