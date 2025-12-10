# ------------------------------------------------------------
# TD0 – Time-Domain Inspection of MLA Signal
#
# Description:
# This file provides an initial inspection of the MLA waveform
# in the time domain. Before analysing resonance frequency or
# performing frequency-domain processing (A3–A9), it is necessary
# to verify that the underlying waveform is clean, stable, and
# free from distortion. TD0 helps identify issues such as:
#
#   • Clipping due to excessive DAC amplitude
#   • Weak excitation (low RMS)
#   • High noise or jitter
#   • Irregular oscillation shape
#   • Missing or unreliable zero-crossings
#
# The file visualises:
#   1. Raw MLA signal (first portion)
#   2. Cleaned signal after DC removal
#   3. A zoomed segment to inspect sinusoidal shape
#   4. Zero-crossing based frequency estimate
#   5. Cycle-to-cycle period jitter
#   6. Histogram of amplitude distribution
#
# Equations used:
#
# 1. DC removal:
#       x_clean[n] = x[n] − mean(x)
#
# 2. RMS amplitude:
#       RMS = sqrt( (1/N) * Σ x[n]^2 )
#
# 3. Zero-crossing frequency:
#       Zero-crossings occur when x[n] < 0 and x[n+1] ≥ 0
#       period[i] = (ZC[i+1] − ZC[i]) / fs
#       f_td = 1 / mean(period)
#
# 4. Jitter:
#       jitter[i] = period[i] − mean_period
#       jitter_ns = jitter * 1e9
#
# These reflect the time-domain stability checks required before
# performing MLA frequency extraction and overtone tracking.
#
# ------------------------------------------------------------

import numpy as np
import sys, json, os
import matplotlib.pyplot as plt

print("\n--------------------------------------------------")
print("TD0 –Time-Domain Inspection of MLA Signal")
print("--------------------------------------------------\n")

# ------------------------------------------------------------
# Print equations again for convenience
# ------------------------------------------------------------

print("\nEquations (Reference)")
print("---------------------")
print("DC removal: x_clean[n] = x[n] − mean(x)")
print("RMS: sqrt((1/N) * Σ x[n]^2)")
print("Zero-crossing frequency: 1 / mean_period")
print("Jitter: (period − mean_period) converted to ns")

# ------------------------------------------------------------
# === FILES / MODES ===
# ------------------------------------------------------------
SIM_MODE             = True
NEW_PYTHON_FILE_MODE = False
OLD_PYTHON_FILE_MODE = False

RAW_FILE   = "mla_data.raw"
META_FILE  = "mla_data.json"

MIN_ZC     = 10
SKIP_ZC    = 2
CSV_A5     = "a5_results.csv"
peak_hz_refined = None

T_View = 300e-6   # 300 microseconds

print(f"[TD1A] ok | SIM_MODE={SIM_MODE} | T_View={T_View} seconds")

# ------------------------------------------------------------
# STEP 1B – Load or generate x and fs
# ------------------------------------------------------------
if SIM_MODE:
    fs = 125_000_000
    N  = 50_000
    t  = np.arange(N) / fs
    f0 = 15_000_000
    x  = np.sin(2*np.pi*f0*t)

    print(f"[TD1B] SIM | fs={fs}, N={N}, f0={f0/1e6:.2f} MHz")

else:
    if not NEW_PYTHON_FILE_MODE and not OLD_PYTHON_FILE_MODE:
        raise ValueError("Enable NEW or OLD MLA mode!")

    if NEW_PYTHON_FILE_MODE:
        print("[TD1B] NEW MLA MODE → RAW + JSON")

        if not os.path.exists(RAW_FILE):
            sys.exit("ERROR: Missing mla_data.raw")

        # metadata
        if os.path.exists(META_FILE):
            with open(META_FILE, "r") as f:
                meta = json.load(f)
            fs = float(meta.get("sample_rate_hz", 125_000_000))

        else:
            print("[TD1B WARN] Metadata missing → fs=125 MHz fallback")
            print("[FIX] See A-Series Documentation → metadata issues.\n")
            fs = 125_000_000

        # raw
        with open(RAW_FILE, "rb") as f:
            x = np.frombuffer(f.read(), dtype=np.int16).astype(np.float64)

        if len(x) < 2000:
            print("[TD1B WARN] Raw file extremely short.")
            print("[FIX] Ensure correct samples_to_collect in test_new.py\n")

        x = x - np.mean(x)

    elif OLD_PYTHON_FILE_MODE:
        print("[TD1B] OLD MLA MODE → RAW only")

        if not os.path.exists(RAW_FILE):
            sys.exit("ERROR: Missing mla_data.raw")

        with open(RAW_FILE, "rb") as f:
            x = np.frombuffer(f.read(), dtype=np.int16).astype(np.float64)

        fs = 125_000_000
        x  = x - np.mean(x)

t = np.arange(len(x)) / fs
print(f"[TD1B] Loaded signal | fs={fs} Hz | N={len(x)} samples")

# ------------------------------------------------------------
# STEP 2 – Raw signal plot
# ------------------------------------------------------------
plt.figure()
plt.plot(t[:2000] * 1e6, x[:2000])
plt.xlabel("Time (µs)")
plt.ylabel("Amplitude")
plt.title("TD0: Raw time-domain signal")
plt.grid(True)
plt.savefig("td_signal_overview.png", dpi=150)
plt.show()

# ------------------------------------------------------------
# STEP 3 – Remove DC offset (Plotting clean signal)
# ------------------------------------------------------------
x_dc = np.mean(x)
x    = x - x_dc

if abs(x_dc) > 0.05:
    print(f"[TD3 WARN] High DC offset detected: {x_dc:+.4f}")
    print("[FIX] See A-Series Documentation → DC offset issues.\n")

print(f"[TD3] Normalized amplitude: max={np.max(np.abs(x)):.3f}")

plt.figure()
plt.plot(t[:2000] * 1e6, x[:2000])
plt.xlabel("Time (µs)")
plt.ylabel("Amplitude")
plt.title("TD0: Cleaned time-domain signal")
plt.grid(True)
plt.savefig("td_signal_cleaned.png", dpi=150)
plt.show()

# zoom window
plt.figure()
plt.plot((np.arange(600)/fs) * 1e6, x[:600], linewidth=0.7)
plt.xlabel("Time (µs)")
plt.ylabel("Amplitude")
plt.title("TD0: Zoomed-in signal")
plt.grid(True)
plt.savefig("td_signal_zoom.png", dpi=150)
plt.show()

# ------------------------------------------------------------
# STEP 4 – Rough f0 using zero-crossings
# ------------------------------------------------------------
sgn = np.sign(x)
zc  = np.where((sgn[:-1] < 0) & (sgn[1:] > 0))[0]

if zc.size < 5:
    print("[TD4 WARN] Too few zero-crossings → unstable oscillation?")
    print("[FIX] See A-Series Documentation → Zero-crossing failures.\n")

if zc.size >= 2:
    periods_s = np.diff(zc) / fs
    if periods_s.size > 8:
        periods_s = periods_s[3:-3]
    T_mean    = np.mean(periods_s)
    f0_td_est = 1.0 / T_mean
    print(f"[TD4] f0_td ≈ {f0_td_est/1e6:.3f} MHz")
else:
    print("[TD4] Cannot estimate f0 (not enough ZC).")

# ------------------------------------------------------------
# STEP 5 – Basic time-domain health checks
# ------------------------------------------------------------
dc_offset  = float(np.mean(x))
rms        = float(np.sqrt(np.mean(x**2)))
hit_top    = bool(np.any(x >= 1.0))
hit_bottom = bool(np.any(x <= -1.0))
clipped    = hit_top or hit_bottom

if clipped:
    print("[TD5 WARN] CLIPPING DETECTED (±1 range exceeded)")
    print("[FIX] See A-Series Documentation → DAC amplitude tuning.\n")

if rms < 0.01:
    print("[TD5 WARN] Very weak signal (RMS too low).")
    print("[FIX] Possible low driving amplitude.\n")

if rms > 0.8:
    print("[TD5 WARN] Extremely strong signal (RMS unusually high).")
    print("[FIX] Possible gain error or saturation.\n")

plt.figure()
plt.hist(x, bins=60, density=True)
plt.xlabel("Amplitude")
plt.ylabel("Count")
plt.title("TD0: amplitude histogram")
plt.grid(True)
plt.savefig("td_hist.png", dpi=150)
plt.show()

# ------------------------------------------------------------
# STEP 6 – Zero-crossing refined f0
# ------------------------------------------------------------
sgn = np.sign(x)
zc2 = np.where((sgn[:-1] <= 0) & (sgn[1:] > 0))[0]
zc_use = zc2[SKIP_ZC:]

if zc_use.size < MIN_ZC:
    print(f"[TD6 WARN] Only {zc_use.size} usable ZCs → inaccurate f0.")
    print("[FIX] See A-Series Documentation → ZC tuning.\n")
else:
    periods_s = np.diff(zc_use) / fs
    T_mean    = np.mean(periods_s)
    f0_td     = 1.0 / T_mean
    print(f"[TD6] f0_td = {f0_td/1e6:.6f} MHz")

# ------------------------------------------------------------
# STEP 7 – Compare TD vs FFT (A5)
# ------------------------------------------------------------
if os.path.exists(CSV_A5):
    with open(CSV_A5, "r") as f:
        header = f.readline().strip().split(",")
        row    = f.readline().strip().split(",")
        j = header.index("peak_freq_hz")
        peak_hz_refined = float(row[j])

if peak_hz_refined is None:
    print("[TD7 WARN] Cannot compare with A5 → CSV missing.")
    print("[FIX] Run A5 on this file before TD0.\n")
else:
    if 'f0_td' in locals() and np.isfinite(f0_td):
        f_diff  = f0_td - peak_hz_refined
        ppm_err = (f_diff / peak_hz_refined) * 1e6
        print(f"[TD7] TD–FFT diff: {f_diff:.1f} Hz ({ppm_err:.3f} ppm)")
    else:
        print("[TD7 WARN] TD estimate missing → cannot compare.")

# ------------------------------------------------------------
# STEP 8 – Jitter distribution
# ------------------------------------------------------------
if "periods_s" in locals():
    jitter_ns = periods_s * 1e9

    if np.std(jitter_ns) > 10:
        print("[TD8 WARN] Large cycle jitter (>10 ns).")
        print("[FIX] See A-Series Documentation → jitter causes.\n")

    plt.figure()
    plt.hist(jitter_ns, bins=50)
    plt.xlabel("Period [ns]")
    plt.ylabel("Count")
    plt.title("TD0: Cycle-to-cycle Jitter")
    plt.savefig("td8_period_jitter.png", dpi=150)
    plt.show()

else:
    print("[TD8] No jitter data available.")

# ------------------------------------------------------------
# SAVE TD RESULTS TO CSV
# ------------------------------------------------------------

import csv
import numpy as np

print("Saving TD results to td_results.csv ...")

# Collect available data safely
csv_data = {}
try:
    csv_data["time_us"] = t * 1e6
except:
    pass

try:
    csv_data["signal"] = x
except:
    pass

try:
    csv_data["cleaned_signal"] = x_clean
except:
    pass

try:
    csv_data["envelope"] = envelope
except:
    pass

try:
    csv_data["periods_ns"] = periods * 1e9
except:
    pass

# Find the longest array length
max_len = max(len(v) for v in csv_data.values())

# Pad arrays so all columns have equal length
for key in csv_data:
    arr = csv_data[key]
    if len(arr) < max_len:
        pad_len = max_len - len(arr)
        csv_data[key] = np.pad(arr, (0, pad_len), constant_values=np.nan)

# Write CSV
with open("td_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(csv_data.keys())  # header row
    writer.writerows(zip(*csv_data.values()))

print("Saved: td_results.csv")


print("TD0 –Time-Domain Inspection of MLA Signal Completed")

