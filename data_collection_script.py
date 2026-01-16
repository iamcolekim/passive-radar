import subprocess
import threading
import time
from datetime import datetime

# ---- USER INPUT ----
freq_mhz = float(input("Enter frequency (MHz): "))
duration = float(input("Enter duration (seconds): "))

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
    "-l", "24",
    "-g", "50",
    "-r", out2,
    "-f", str(frequency),
    "-s", "20000000"
]

cmd_slave = [
    "hackrf_transfer",
    "-H",
    "-d", SERIAL_SLAVE,
    "-a", "0",
    "-l", "24",
    "-g", "50",
    "-r", out1,
    "-f", str(frequency),
    "-s", "20000000"
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

p1 = run_hackrf(cmd_slave, log1)
time.sleep(0.2)  # ensure slave starts first
p2 = run_hackrf(cmd_master, log2)

# ---- RUN FOR DURATION ----
time.sleep(duration)

# ---- STOP RECORDING ----
print("Stopping HackRF recordings...")

p1.terminate()
p2.terminate()

p1.wait()
p2.wait()

print("Done.")



