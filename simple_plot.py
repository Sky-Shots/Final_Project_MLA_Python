
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# --- FILE MODE SELECTOR ---
USE_NEW_TEST = False
if USE_NEW_TEST:
    RAW_FILE  = "mla_data.raw"
    META_FILE = "mla_data.json"
else:
    RAW_FILE  = "mla_data2.raw"
    META_FILE = None


SAMPLE_RATE = 125_000_000

# ---- Load metadata only in New Mode ---

if USE_NEW_TEST and META_FILE is not None:
    import json
    try:
        with open(META_FILE, "r") as f:
            meta = json.load(f)
        print("\nMetadta loaded sucessfully.")
        print("Sample rate:", meta.get("sample_rate_hz","Missing"))
    except Exception as e:
        print("Warning: Could not load JSON metadata:", e)
        meta = None
else:
    meta = None


# load your data
# Load data from the raw binary file
file_path = RAW_FILE
try:
    # Read the binary data as signed 16-bit integers
    data = np.fromfile(file_path, dtype=np.int16)
    data = data / 8192
    print(f"Loaded data with shape: {data.shape}")
except FileNotFoundError:
    print(f"Error: '{file_path}' not found. Please make sure the file exists.")
    exit(1)


# Remove DC offset
data = data - np.mean(data)
print("Removed DC offset.")

t = np.arange(len(data)) / SAMPLE_RATE * 1000  # Time vector in milliseconds
# Highpass filter > 1 MHz
b, a = signal.butter(4, 1e6, 'high', fs=SAMPLE_RATE)
data = signal.filtfilt(b, a, data)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Time domain plot on the first subplot
ax1.plot(t, data)
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('Amplitude (V)')
ax1.set_title('MLA Signal (Time Domain)')
ax1.grid(True)

# Compute FFT
n = len(data)
fft_data = np.fft.rfft(data)
fft_freq = np.fft.rfftfreq(n, d=1/SAMPLE_RATE)
# Convert frequency to MHz for better readability
fft_freq_MHz = fft_freq / 1e6

# Frequency domain plot on the second subplot
ax2.plot(fft_freq_MHz, np.abs(fft_data))
ax2.set_xlabel('Frequency (MHz)')
ax2.set_ylabel('Magnitude')
ax2.set_title('Frequency Spectrum')
ax2.set_xlim(0, 61)  # Limit to 61 MHz for better visibility
ax2.grid(True)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
