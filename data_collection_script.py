import subprocess
import threading
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------
# Settings
# -------------------------------------------------
NUM_SAMPLES = 1_000_000   # first 1M complex samples
PLOT_SAMPLES = 2000

# -------------------------------------------------
# Helpers
# -------------------------------------------------

def print_stats(name, s):
    print(f"\n{name}")
    print(f"  Total complex samples : {s['total_complex']:,}")
    print(f"  I clipped samples     : {s['I_clipped']:,}")
    print(f"  Q clipped samples     : {s['Q_clipped']:,}")
    print(f"  Complex clipped       : {s['complex_clipped']:,}")
    print(f"  % complex clipped     : {s['percent_complex_clipped']:.3f}%")

def read_cs8_first_n_with_stats(path, n_complex):
    """
    Read first n_complex samples from cs8 (int8 IQ interleaved).
    Returns:
      x_complex : complex64
      stats     : dict with clipping information
    """
    raw = np.fromfile(path, dtype=np.int8, count=2 * n_complex)
    raw = raw[: raw.size - (raw.size % 2)]

    I = raw[0::2]
    Q = raw[1::2]

    # Clipping detection
    I_clip = (I == 127) | (I == -128)
    Q_clip = (Q == 127) | (Q == -128)
    complex_clip = I_clip | Q_clip

    stats = {
        "total_complex": len(I),
        "I_clipped": int(np.sum(I_clip)),
        "Q_clipped": int(np.sum(Q_clip)),
        "complex_clipped": int(np.sum(complex_clip)),
        "percent_complex_clipped": 100.0 * np.mean(complex_clip)
    }

    x = I.astype(np.float32) + 1j * Q.astype(np.float32)
    return x, stats

def normalize(x):
    x = x - np.mean(x)
    rms = np.sqrt(np.mean(np.abs(x) ** 2))
    return x / rms if rms > 0 else x

def cross_correlate_fft(x, y):
    n = len(x)
    m = len(y)
    L = n + m - 1
    nfft = 1 << (L - 1).bit_length()

    X = np.fft.fft(x, nfft)
    Y = np.fft.fft(y, nfft)
    corr = np.fft.ifft(X * np.conj(Y))
    corr = np.concatenate((corr[-(m-1):], corr[:n]))
    lags = np.arange(-(m - 1), n)
    return corr, lags

def align_signals(x, y, lag):
    if lag > 0:
        x = x[lag:]
        y = y[:len(x)]
    elif lag < 0:
        y = y[-lag:]
        x = x[:len(y)]
    n = min(len(x), len(y))
    return x[:n], y[:n]





# ---- USER INPUT ----
freq_mhz = float(input("Enter frequency (MHz): "))
duration = float(input("Enter duration (seconds): "))
samp_rate = float(input("Enter sampling rate (MSps): "))
gain_lna = input("Enter LNA gain (dB, steps of 8): ")
if (gain_lna == ""): 
    gain_lna = "16"
gain_vga = input("Enter VGA gain (db, steps of 2): ")
if (gain_vga == ""):
    gain_vga = "16"

print("gain settings: lna = ", gain_lna,", vga =", gain_vga, "\n")
# Convert to Hz
frequency = int(freq_mhz * 1_000_000)

# Timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Output filenames
out1 = f"63_{freq_mhz}MHz_{timestamp}.cs8"
out2 = f"9f_{freq_mhz}MHz_{timestamp}.cs8"

log1 = f"63_{freq_mhz}MHz_{timestamp}.log"
log2 = f"9f_{freq_mhz}MHz_{timestamp}.log"

# HackRF serial numbers
SERIAL_MASTER = "000000000000000066a062dc226b169f"
SERIAL_SLAVE  = "0000000000000000436c63dc38284c63"

# HackRF commands
cmd_master = [
    "hackrf_transfer",
    "-d", SERIAL_MASTER,
    "-a", "0",
    "-l", gain_lna,
    "-g", gain_vga,
    "-r", out2,
    "-f", str(frequency),
    "-s", str(samp_rate * 1_000_000)
]

cmd_slave = [
    "hackrf_transfer",
    "-H",
    "-d", SERIAL_SLAVE,
    "-a", "0",
    "-l", gain_lna,
    "-g", gain_vga,
    "-r", out1,
    "-f", str(frequency),
    "-s", str(samp_rate * 1_000_000)
]

clk_master = [
    "hackrf_clock",
    "-d", SERIAL_MASTER,
    "-o", "1"
]

clk_slave = [
    "hackrf_clock",
    "-d", SERIAL_SLAVE,
    "-i"
]
# ---- PROCESS HANDLER ----
def run_hackrf(cmd, logfile):
    with open(logfile, "w") as log:
        return subprocess.Popen(
            cmd,
            stdout=log,
            stderr=log,
            text=True
        )


# ---- START RECORDING ----
print("Starting HackRF recordings...")

p0 = run_hackrf(clk_master, log2)
time.sleep(0.2)
p1 = run_hackrf(clk_slave, log1)
time.sleep(0.2)
p2 = run_hackrf(cmd_slave, log1)
time.sleep(0.2)  # ensure slave starts first
p3 = run_hackrf(cmd_master, log2)

# ---- RUN FOR DURATION ----
time.sleep(duration)

# ---- STOP RECORDING ----
print("Stopping HackRF recordings...")

p0.terminate()
p1.terminate()
p2.terminate()
p3.terminate()

p0.wait()
p1.wait()
p2.wait()
p3.wait()

print("Loading first 1M samples + clipping stats...")

x_raw, stats_x = read_cs8_first_n_with_stats(out1 , NUM_SAMPLES)
y_raw, stats_y = read_cs8_first_n_with_stats(out2 , NUM_SAMPLES)


print_stats("File X", stats_x)
print_stats("File Y", stats_y)

# Normalize AFTER stats
x = normalize(x_raw)
y = normalize(y_raw)

print("\nRunning cross-correlation...")
corr, lags = cross_correlate_fft(x, y)
mag = np.abs(corr)

peak_idx = np.argmax(mag)
best_lag = int(lags[peak_idx])

print(f"\nPeak correlation lag: {best_lag} samples")
print(f"Peak |corr|: {mag[peak_idx]:.4f}")

# Align
xa, ya = align_signals(x, y, best_lag)

rho = np.mean(xa * np.conj(ya))
print(f"Aligned correlation coefficient: {rho:.4f} (|ρ|={np.abs(rho):.4f})")

# -------------------------------------------------
# Plot
# -------------------------------------------------
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(lags, mag)
plt.axvline(best_lag, linestyle="--")
plt.title("Cross-correlation magnitude")
plt.xlabel("Lag (samples)")
plt.ylabel("|corr|")
plt.grid(True)

plt.subplot(2, 1, 2)
n = min(PLOT_SAMPLES, len(xa))
plt.plot(np.real(xa[:n]), label="Re(x) aligned")
plt.plot(np.real(ya[:n]), label="Re(y) aligned", alpha=0.8)
plt.title("Time-domain alignment (real part)")
plt.xlabel("Sample")
plt.ylabel("Amplitude (normalized)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("Done.")







