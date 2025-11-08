#!/usr/bin/env python3
"""
NanoVNA + MotorController antenna pattern sweep (JOG mode: small relative moves only)

- No absolute targets; every move is a small relative jog (default +5°)
- One NanoVNA sweep per jog; repeats ~360° and then unwinds back to start
- Logs S11/S21 complex and dB at each step

CSV columns:
  step_idx, nominal_angle_deg, freq_Hz, S11_re, S11_im, S21_re, S21_im, S11_dB, S21_dB
"""

import csv
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pynanovna

# ----------------- USER KNOBS -----------------
MOTOR_PORT       = "/dev/ttyACM0"     # e.g., "COM5" on Windows
MOTOR_BAUD       = 115200
JOG_DEG          = 5.0                # relative nudge per step (+ = forward)
N_STEPS          = None               # None => round(360/JOG_DEG); or set an int
SETTLE_S         = 0.15               # short pause after each jog
CSV_PATH         = Path("nanovna_pattern_1to3GHz_jog.csv")

SWEEP_START_HZ   = 1_000_000_000
SWEEP_STOP_HZ    = 3_000_000_000
SWEEP_POINTS     = 101                # most NanoVNA firmwares cap at 101; we’ll fallback if needed

# -------------- PROJECT DRIVERS --------------
from MotorController import MotorController  # your GRBL-like driver


# ----------------- HELPERS -------------------
def db20(x):
    x = np.asarray(x)
    return 20.0 * np.log10(np.clip(np.abs(x), 1e-12, None))


def configure_vna_with_fallback(vna, start_hz: int, stop_hz: int, points: int) -> int:
    """
    Try desired sweep points; if device rejects, fall back to common safe values.
    Return the configured point count.
    """
    for p in (points, 201, 101, 51):
        try:
            vna.set_sweep(start_hz, stop_hz, int(p))
            # quick validation sweep
            s11, s21, freqs = vna.sweep()
            if len(freqs) and len(s11) == len(freqs) and len(s21) == len(freqs):
                return int(p)
        except Exception:
            continue
    raise RuntimeError("NanoVNA validation failed (points / data).")


def jog_move(mc: MotorController, delta_deg: float, retry: bool = True) -> bool:
    """
    Perform a relative jog via rotate_mast(delta). Return True on success.
    """
    ok = mc.rotate_mast(float(delta_deg))
    if ok:
        return True
    if retry:
        time.sleep(0.25)
        return bool(mc.rotate_mast(float(delta_deg)))
    return False


# ------------------- MAIN --------------------
def main():
    # Connect motor (no homing/absolute mode needed for jog operation)
    mc = MotorController(MOTOR_PORT, MOTOR_BAUD)
    if not mc.connect():
        print("ERROR: Motor controller failed to connect.", file=sys.stderr)
        return 1
    print("Motor connected (JOG mode).")

    # Connect NanoVNA and validate BEFORE any motion
    try:
        vna = pynanovna.VNA()  # auto-detect
        cfg_pts = configure_vna_with_fallback(vna, SWEEP_START_HZ, SWEEP_STOP_HZ, SWEEP_POINTS)
        print(f"NanoVNA OK @ {cfg_pts} points.")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        try: mc.disconnect()
        except Exception: pass
        return 1

    # Decide steps
    jog = float(JOG_DEG if JOG_DEG != 0 else 5.0)
    steps = int(round(360.0 / jog)) if N_STEPS is None else int(N_STEPS)
    steps = max(1, steps)

    # Prepare CSV
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    f = CSV_PATH.open("w", newline="")
    writer = csv.writer(f)
    writer.writerow([
        "step_idx", "nominal_angle_deg", "freq_Hz",
        "S11_re", "S11_im", "S21_re", "S21_im",
        "S11_dB", "S21_dB"
    ])

    # Nominal (for logging only): we don’t rely on absolute position
    nominal_angle = 0.0
    total_jogged = 0.0

    try:
        print(f"Starting jog sweep: {steps} steps × {jog:.1f}° ≈ {steps*jog:.1f}° total")
        for i in range(steps):
            # Relative jog; don’t care about absolute angle
            if not jog_move(mc, jog):
                print(f"[WARN] jog step {i+1}/{steps} failed; recording zeros and continuing.")
                # Still increment nominal so the log stays monotonic
                nominal_angle = (nominal_angle + jog) % 360.0
                total_jogged += jog
                # Write “empty” rows for this step to keep CSV shape predictable?
                # We’ll just skip sweep here to save time.
                continue

            total_jogged += jog
            time.sleep(SETTLE_S)
            nominal_angle = (nominal_angle + jog) % 360.0

            # Sweep & log
            s11, s21, freqs = vna.sweep()
            if not len(freqs):
                print(f"[WARN] empty VNA sweep at step {i} — skipping log.")
                continue

            s11 = np.asarray(s11, dtype=np.complex128)
            s21 = np.asarray(s21, dtype=np.complex128)
            s11_db = db20(s11)
            s21_db = db20(s21)

            for k in range(len(freqs)):
                writer.writerow([
                    i, f"{nominal_angle:.1f}", int(freqs[k]),
                    f"{s11[k].real:.9e}", f"{s11[k].imag:.9e}",
                    f"{s21[k].real:.9e}", f"{s21[k].imag:.9e}",
                    f"{s11_db[k]:.6f}", f"{s21_db[k]:.6f}"
                ])
            f.flush()
            print(f"Step {i+1:3d}/{steps} (nom≈{nominal_angle:6.1f}°) → wrote {len(freqs)} pts")

        # Unwind back to start (best-effort)
        if abs(total_jogged) > 0.01:
            print(f"Unwinding by {-total_jogged:.2f}° to return near start …")
            # split unwind into safe chunks in case firmware limits single-move size
            remaining = -total_jogged
            while abs(remaining) > 0.01:
                chunk = max(min(remaining, 45.0), -45.0)  # ±45° chunks
                if not jog_move(mc, chunk, retry=False):
                    # if an unwind chunk fails, just break; we’re in jog mode anyway
                    break
                remaining -= chunk
                time.sleep(0.05)

        print(f"\nDone. CSV saved to: {CSV_PATH.resolve()}")

    finally:
        try: f.close()
        except Exception: pass
        try: vna.close()
        except Exception: pass
        try: mc.disconnect()
        except Exception: pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
