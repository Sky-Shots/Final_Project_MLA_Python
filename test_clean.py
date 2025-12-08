"""
test.py — MLA Data Acquisition Script for QCM (Red Pitaya FPGA)

This script performs one complete MLA measurement using the custom FPGA
bitstream on the Red Pitaya. It is written clearly and simply so anyone
can understand the workflow without needing to read the FPGA code.

The script does the following:

1. Connects to FPGA registers (AXI-Lite) and DDR memory
2. Resets the FPGA and prepares it for a new measurement
3. Generates the MLA frequency-comb waveform (two’s-complement DAC)
4. Pads the waveform to 64-sample alignment (required for DMA)
5. Writes the waveform into DDR RAM at address 0
6. Sets up the FPGA DMA reader and writer regions
7. Starts the measurement by releasing reset
8. Waits until the FPGA finishes writing captured ADC data
9. Reads the captured data from RAM into mla_data.raw
10. Saves measurement information into mla_data.json
11. Resets and closes the device safely

This file only captures data.
All signal analysis (A3–A9, TD-series) is done separately.

This version keeps the *exact* working logic of the original test.py,
but reorganizes it for clarity, safety, and readability.
"""



# --- Imports ----

""" MMIO  --> gives access to FPGA registers and DDR RAM
    NUMPY --> for generating waveform and array handling
    time  --> for delays and safe polling
    json  --> for writing measumerment meta data
    sys   --> for safe exit an errors
    Frequency-comb generator 
"""
from periphery import MMIO
import numpy as np
import struct
import json
import time
import sys
from freq_comb_gen import freq_comb_twos_comp_dac


# --- CONFIGURATION — User adjustable parameters ---

# MLA excitation settings
CENTER_FREQUENCIES = [14.85e6]   # center frequencies in Hz
DESIRED_BANDWIDTH  = 500         # target bandwidth in Hz
COMB_SIZE          = 33          # number of tones per comb (odd number)
AMPLITUDE_PEAK_V   = 0.15        # DAC peak amplitude in volts

# Capture settings
REPEAT_CYCLES      = 2048        # how many times to repeat the waveform in RAM
CLEAR_RAM_BEFORE   = True        # clear DDR RAM before writing waveform
ENABLE_RELAY       = True        # connect excitation to QCM crystal
ENABLE_TRIGGER     = False       # external trigger (usually OFF)
READER_CONTINUOUS  = True        # reader mode (continuous or one-shot)

# Output files (required by A-series and TD-series)
RAW_OUTPUT_FILE  = "mla_data.raw"
JSON_OUTPUT_FILE = "mla_data.json"


# --- FPGA Register Map (from the working bitstream)
# Each register is 32 bit . Flags normally use only bit 0.

# Identity register used to confrim we are connected to the MLA bitstrem

MLA_CONST_REG= 0X00

# Write(stimulus --> DAC --> FPGA) configuation registers

WRITER_MIN_ADDRESS_REG      = 0X04      # DDR start address for writer
WRITER_NUM_BURSTS_REG       = 0X08      # Number of 128-bytes bursts
WRITE_WRIT_PTR_REG          = 0X10      # Writer pointer (status)
WRITER_DONE_REG             = 0X14      # Writer finished (bit 0)

# Reader (ADC --> FPGA --> DDR RAM) configuration registers

READER_MIN_ADDRESS_REG      = 0X20      # DDR Start Address For Reader
READER_READ_PTR_REG         = 0X24      # Reader pointer (status)
READER_DONE_REG             = 0X28      # Reader finished (bit 0)


# Reader/ Writer  comaprison counters(status only)
READER_COMP_BURST_OUT_REG   = 0X2C
WRITER_COMP_BURTS_OUT_REG   = 0x34

# General timing & Control

RELAXATION_TIME_REG         = 0X18
EXCITATION_TIME_REG         = 0X1C
READER_NUM_BURSTS_REG       = 0X30
READER_CONTINUOUS_MODE_REG  = 0X38
WRITER_CONTINUOUS_MODE_REG  = 0X3C

# Output Control (Realy + Tigger)

RELAY_ENABLED_REG           = 0X40
TRIGGER_ENABLED_REG         = 0X44

# --- DDR RAM Base address and total size ---

RAM_BASE = 0X1000000                    # Start of DDR buffer
RAM_SIZE = 190 * 1024 * 1024            # 190 MB available

# --- General Constants ---

SAMPLE_RATE         = 125_000_000       # 125 MHz ADC sampling rate
SAMPLE_SIZE         = 2                 # int16 = 2 bytes
BURST_SIZE_BYTES    =128                # DDR bursts = 128 bytes

# --- STEP 4 - MAP FPGA AXI -Lite and DDR memory ---

print("Mapping FPGA memory regions...")

try:
    axil_mmio = MMIO(AXIL_BASE := 0X40000000 , AXIL_SIZE := 0X1000)
    ram_mmio  = MMIO(RAM_BASE, RAM_SIZE)
except Exception as e :
    print(f"ERROR : Unable to map FPGA memory regions: {e}")
    sys.exit(1)
    
print("Memmory mapping successful. ")


# --- STEP 5 — Sanity Check: Confirm FPGA Identity ---

print("Checking FPGA identity...")

raw_id = axil_mmio.read32(MLA_CONST_REG)

# The FPGA writes ASCII characters (e.g., "MLA") into this register.
id_str = raw_id.to_bytes(4, byteorder="big").decode("latin-1", errors="ignore").strip()

if id_str != "MLA":
    print(f"WARNING: Unexpected FPGA ID '{id_str}' (raw=0x{raw_id:08X})")
else:
    print(f"FPGA identity OK: '{id_str}'")



# STEP 6 — Reset the Device Before Measurement


print("Resetting FPGA...")

# Soft reset (active-low, so writing 1 holds system in reset)
axil_mmio.write32(SOFT_RESET_REG, 1)

# Clear writer configuration
axil_mmio.write32(WRITER_MIN_ADDRESS_REG, 0)
axil_mmio.write32(WRITER_NUM_BURSTS_REG, 0)

# Clear reader configuration
axil_mmio.write32(READER_MIN_ADDRESS_REG, 0)
axil_mmio.write32(READER_NUM_BURSTS_REG, 0)

# Disable modes and outputs
axil_mmio.write32(RELAY_ENABLED_REG, 0)
axil_mmio.write32(READER_CONTINUOUS_MODE_REG, 0)
axil_mmio.write32(WRITER_CONTINUOUS_MODE_REG, 0)
axil_mmio.write32(TRIGGER_ENABLED_REG, 0)

time.sleep(0.4)   # allow hardware to settle

print("FPGA reset complete. Device is stable.")

# ---------------------------------------------------------
# STEP 6 — Generate MLA Frequency Comb (using top-level knobs)
# ---------------------------------------------------------

print("Generating MLA excitation waveform...")

# Use the configuration values defined at the top of the file
center_freqs_des = CENTER_FREQUENCIES
desired_bw_des   = DESIRED_BANDWIDTH
comb_sz          = COMB_SIZE
amplitude_pk_v   = AMPLITUDE_PEAK_V

# Generate waveform using the same function as original test.py
comb_waveform_tc, t_vec, actual_bw, n_samples, actual_freqs = freq_comb_twos_comp_dac(
    center_freqs_des,
    desired_bw_des,
    comb_sz,
    amplitude_pk_v,
    apply_t_shift=True
)

# Report results
print(f"Waveform generated with {n_samples} samples.")
print(f"Actual tone frequencies (MHz): {actual_freqs / 1e6}")
print(f"Actual bandwidth (Hz): {actual_bw}")


# --- STEP 6.1 — Pad waveform to 64-sample alignment ---


# DDR DMA requires the number of samples to be a multiple of 64
pad = (-len(comb_waveform_tc)) % 64
if pad > 0:
    comb_waveform_tc = np.pad(comb_waveform_tc, (0, pad), mode="constant")
    n_samples = len(comb_waveform_tc)
    print(f"Waveform padded by {pad} samples → new length = {n_samples}")


# --- STEP 7 — Compute RAM Offsets and Write Waveform to DDR ---


# Size of waveform in bytes (2 bytes per int16 sample)
waveform_num_bytes = len(comb_waveform_tc) * SAMPLE_SIZE

# Number of 128-byte bursts needed to store waveform
waveform_num_bursts = waveform_num_bytes // BURST_SIZE_BYTES

print(f"[INFO] Waveform size : {waveform_num_bytes} bytes")
print(f"[INFO] Burst count   : {waveform_num_bursts} bursts")


# Writer offset = where ADC captured data will be stored
# This MUST NOT overlap the waveform stored at RAM_BASE.


writer_offset = RAM_BASE + waveform_num_bytes     # absolute address in RAM

# FPGA expects relative addressing starting at 0 for DDR
writer_offset_rel = writer_offset - RAM_BASE      # critical for correct readback

# Maximum writable space after waveform, in bytes
writer_limit = RAM_SIZE - waveform_num_bytes

# The FPGA writer stores data in fixed 128-byte bursts.
# We limit it based on repeat cycles as in the original working script.
writer_num_bursts = min(
    REPEAT_CYCLES * waveform_num_bursts,
    writer_limit // BURST_SIZE_BYTES
)

# Total bytes the writer will actually store
true_writer_bytes = writer_num_bursts * BURST_SIZE_BYTES

print(f"[INFO] Writer offset (abs) : 0x{writer_offset:X}")
print(f"[INFO] Writer offset (rel) : 0x{writer_offset_rel:X}")
print(f"[INFO] Writer bursts       : {writer_num_bursts}")
print(f"[INFO] Capture size        : {true_writer_bytes} bytes")


# STEP 7.1 — Optional: Clear RAM before writing


if CLEAR_RAM_BEFORE:
    print("Clearing DDR RAM (this may take a few seconds)...")
    chunk = 1 * 1024 * 1024  # 1 MB chunks
    for addr in range(0, RAM_SIZE, chunk):
        size = min(chunk, RAM_SIZE - addr)
        ram_mmio.write(addr, b"\x00" * size)
    print("RAM cleared.")


# STEP 7.2 — Write waveform to DDR RAM at address 0

print("Writing waveform to RAM...")
ram_mmio.write(0, comb_waveform_tc.tobytes())
print("Waveform written successfully.")

# ---------------------------------------------------------
# STEP 8 — Configure FPGA Reader/Writer Registers
# ---------------------------------------------------------

print("Configuring FPGA registers...")

# Reader configuration (ADC → RAM)
axil_mmio.write32(READER_MIN_ADDRESS_REG, RAM_BASE)
axil_mmio.write32(READER_NUM_BURSTS_REG, waveform_num_bursts)

# Writer configuration (capture → RAM)
axil_mmio.write32(WRITER_MIN_ADDRESS_REG, writer_offset)
axil_mmio.write32(WRITER_NUM_BURSTS_REG, writer_num_bursts)

# Continuous mode settings
axil_mmio.write32(READER_CONTINUOUS_MODE_REG, 1 if READER_CONTINUOUS else 0)
axil_mmio.write32(WRITER_CONTINUOUS_MODE_REG, 0)   # Writer always one-shot

# Relay control (connect excitation to QCM crystal)
axil_mmio.write32(RELAY_ENABLED_REG, 1 if ENABLE_RELAY else 0)

# Trigger control (external trigger usually disabled)
axil_mmio.write32(TRIGGER_ENABLED_REG, 1 if ENABLE_TRIGGER else 0)

# Debug print
print(f"Reader bursts  : {axil_mmio.read32(READER_NUM_BURSTS_REG)}")
print(f"Writer bursts  : {axil_mmio.read32(WRITER_NUM_BURSTS_REG)}")
print(f"Reader address : 0x{axil_mmio.read32(READER_MIN_ADDRESS_REG):X}")
print(f"Writer address : 0x{axil_mmio.read32(WRITER_MIN_ADDRESS_REG):X}")

print("FPGA configuration complete.")

input("Press Enter to start measurement...")  # Pause to let user verify settings

# Start measurement by releasing reset (0 = run)
axil_mmio.write32(SOFT_RESET_REG, 0)
print("Measurement started...")


# ---------------------------------------------------------
# STEP 9 — Wait for FPGA Writer & Reader to Finish
# ---------------------------------------------------------

print("Waiting for FPGA to finish measurement...")

writer_done = 0
reader_done = 0

while not (writer_done & reader_done):

    writer_done = axil_mmio.read32(WRITER_DONE_REG) & 0x1
    reader_done = axil_mmio.read32(READER_DONE_REG) & 0x1

    write_ptr = axil_mmio.read32(WRITER_WRITE_PTR_REG)
    read_ptr  = axil_mmio.read32(READER_READ_PTR_REG)

    print(f"Writer ptr: {write_ptr} | Reader ptr: {read_ptr}", end="\r")
    time.sleep(0.05)

print("\nMeasurement complete.")
print(f"Writer done = {writer_done}, Reader done = {reader_done}")


# ---------------------------------------------------------
# STEP 10 — Read Captured ADC Data from DDR into mla_data.raw
# ---------------------------------------------------------

print("Reading captured data from DDR RAM...")

total_bytes = true_writer_bytes      # exact bytes captured
chunk_size  = 8 * 1024 * 1024        # 8 MB chunks
bytes_read  = 0

print(f"Saving capture to '{RAW_OUTPUT_FILE}'...")
with open(RAW_OUTPUT_FILE, "wb") as f:

    for offset in range(0, total_bytes, chunk_size):

        # Amount to read this iteration
        read_size = min(chunk_size, total_bytes - offset)

        # Read from RELATIVE writer offset (critical!)
        chunk = ram_mmio.read(writer_offset_rel + offset, read_size)

        f.write(chunk)
        bytes_read += read_size

        progress = (bytes_read / total_bytes) * 100
        print(f"\rProgress: {progress:.1f}% ({bytes_read/1024/1024:.1f} MB)", end="")
        del chunk  # free memory

print("\nCapture saved successfully.")



# ---------------------------------------------------------
# STEP 11 — Write JSON metadata file
# ---------------------------------------------------------

print(f"Creating metadata file '{JSON_OUTPUT_FILE}'...")

metadata = {
    "center_frequencies_hz": CENTER_FREQUENCIES,
    "desired_bandwidth_hz": DESIRED_BANDWIDTH,
    "comb_size": COMB_SIZE,
    "amplitude_peak_v": AMPLITUDE_PEAK_V,
    "sample_rate_hz": SAMPLE_RATE,

    # Waveform information
    "waveform_samples": int(n_samples),
    "waveform_bytes": int(waveform_num_bytes),
    "actual_bandwidth_hz": float(actual_bw),
    "actual_frequencies_hz": [float(f) for f in actual_freqs.tolist()],

    # Capture information
    "writer_offset_bytes": int(writer_offset_rel),
    "writer_num_bursts": int(writer_num_bursts),
    "capture_bytes": int(true_writer_bytes),

    # File outputs
    "raw_output_file": RAW_OUTPUT_FILE,
    "json_output_file": JSON_OUTPUT_FILE,

    # Timestamp
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
}

with open(JSON_OUTPUT_FILE, "w") as jf:
    json.dump(metadata, jf, indent=2)

print("Metadata saved.")


# ---------------------------------------------------------
# STEP 12 — Reset device and close memory mappings
# ---------------------------------------------------------

print("Resetting FPGA and closing device...")

# Put FPGA back into reset (safe idle state)
axil_mmio.write32(SOFT_RESET_REG, 1)

# Close memory regions
axil_mmio.close()
ram_mmio.close()

print("Device closed successfully.")
print("Measurement complete. Exiting.")






