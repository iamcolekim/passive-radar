#!/usr/bin/env python3
"""
RAW CS8 I-ONLY PLAYBACK -> GIF

- Reads cs8 (int8 IQ interleaved) blocks from reference + surveillance files
- Extracts ONLY I (real) samples
- Overlays Reference I and Surveillance I on one graph
- Saves animation as GIF (no ffmpeg required)

Dependencies:
  pip install numpy matplotlib pillow
"""

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")  # reliable offscreen backend on macOS
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


# ----------------------------
# CS8 reader (int8 IQ interleaved)
# ----------------------------
def read_cs8_block_I(f, num_complex: int):
    """
    Reads num_complex complex samples from a cs8 file and
    returns ONLY the I (real) part as float32.
    """
    raw = np.fromfile(f, dtype=np.int8, count=2 * num_complex)
    if raw.size < 2 * num_complex:
        return None
    I = raw[0::2].astype(np.float32)
    return I


def main():
    # ----------------------------
    # File paths
    # ----------------------------
    fid_x_path = "/Users/ibrahimsweidan/Downloads/500M/63_500M_01_11_car_2.cs8"  # Reference
    fid_y_path = "/Users/ibrahimsweidan/Downloads/500M/63_500M_01_11_car_2.cs8"  # Surveillance

    if not os.path.exists(fid_x_path) or not os.path.exists(fid_y_path):
        raise FileNotFoundError("Could not find one or both input files.")

    # ----------------------------
    # Parameters
    # ----------------------------
    length = 2**15       # samples per block
    plot_N = 4096        # samples to plot per frame (for clarity/speed)
    max_record = 50      # number of frames
    fps = 20
    gif_filename = "PassiveRadar_Raw_I_Overlay.gif"

    # ----------------------------
    # Read raw I blocks
    # ----------------------------
    print("Recording raw I-only blocks...")

    x_blocks = []
    y_blocks = []

    with open(fid_x_path, "rb") as fx, open(fid_y_path, "rb") as fy:
        for k in range(max_record):
            xI = read_cs8_block_I(fx, length)
            yI = read_cs8_block_I(fy, length)

            if xI is None or yI is None:
                print("\nEOF reached early.")
                break

            x_blocks.append(xI[:plot_N].copy())
            y_blocks.append(yI[:plot_N].copy())

            pct = ((k + 1) / max_record) * 100
            bar_len = 30
            filled = int(round((pct / 100) * bar_len))
            empty = bar_len - filled
            print(
                f"[{'#'*filled}{'-'*empty}] {pct:5.1f}% ({k+1}/{max_record} blocks)",
                end="\r",
                flush=True,
            )

    print("\nRecording complete. Generating GIF...")

    nframes = min(len(x_blocks), len(y_blocks))
    if nframes == 0:
        raise RuntimeError("No frames recorded.")

    # ----------------------------
    # Axis setup
    # ----------------------------
    n = np.arange(plot_N)

    all_vals = np.concatenate(x_blocks + y_blocks)
    vmin = float(all_vals.min()) - 5.0
    vmax = float(all_vals.max()) + 5.0

    # ----------------------------
    # Plot / animation
    # ----------------------------
    fig, ax = plt.subplots(figsize=(7, 7), dpi=90)

    line_ref, = ax.plot([], [], linewidth=1.2, label="Reference I")
    line_surv, = ax.plot([], [], linewidth=1.2, label="Surveillance I")

    ax.set_title("Raw I Component (Reference & Surveillance)")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Amplitude (int8 units)")
    ax.grid(True)
    ax.set_xlim([0, plot_N - 1])
    ax.set_ylim([vmin, vmax])
    ax.legend(loc="upper right")

    def init():
        line_ref.set_data([], [])
        line_surv.set_data([], [])
        return line_ref, line_surv

    def update(i):
        line_ref.set_data(n, x_blocks[i])
        line_surv.set_data(n, y_blocks[i])
        ax.set_title(f"Raw I Component — Block {i+1}/{nframes}")
        return line_ref, line_surv

    anim = FuncAnimation(
        fig,
        update,
        frames=nframes,
        init_func=init,
        blit=True,
        interval=int(1000 / fps),
    )

    writer = PillowWriter(fps=fps)
    anim.save(gif_filename, writer=writer)

    plt.close(fig)
    print(f"Finished! GIF saved to: {gif_filename}")


if __name__ == "__main__":
    main()
