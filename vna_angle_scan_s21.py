#!/usr/bin/env python3
"""
Antenna pattern sweep with NanoVNA + motor controller.

- Rotates 0..355° in 5° steps (absolute moves using MotorController.rotate_mast)
- At each angle: sweep 1–3 GHz and log S11/S21 (complex + |.| in dB) to CSV.
- Safety:
    * Validates NanoVNA returns data before any motion.
    * Verifies each move completed (±0.5°), with one automatic retry.
- Clean shutdown: returns to ~0°, closes CSV and device handles.

Requirements:
    pip install pynanovna pyserial numpy

Notes:
  - Logs S11 and S21 (2-port NanoVNA). S12/S22 usually require a second pass or reversing DUT.
  - Uses your MotorController from MotorController.py.
"""

import csv
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pynanovna  # NanoVNA control

# --- USER SETTINGS ------------------------------------------------------------
MOTOR_PORT      = "/dev/ttyACM0"   # e.g., "COM5" on Windows
MOTOR_BAUD      = 115200
SWEEP_START_HZ  = 1_000_000_000
SWEEP_STOP_HZ   = 3_000_000_000
SWEEP_POINTS    = 101              # 101/201/401 are typical; keep modest
ANGLE_STEP_DEG  = 5.0              # 0,5,...,355 (no 360 duplicate)
SETTLE_S        = 0.20             # pause after each move before measuring
ANGLE_TOL_DEG   = 0.5              # verification tolerance
CSV_PATH        = Path("nanovna_pattern_1to3GHz.csv")

# --- IMPORT YOUR MOTOR CONTROLLER --------------------------------------------
from MotorController import MotorController  # provided by you


# --- SMALL HELPERS ------------------------------------------------------------
def db20(x):
    x = np.asarray(x)
    return 20.0 * np.log10(np.clip(np.abs(x), 1e-12, None))


def ang_err(a, b):
    """Smallest signed angular error a−b in degrees (wrap-aware)."""
    # Map difference to (-180, 180]
    return ((a - b + 180.0) % 360.0) - 180.0


def read_mast_deg(mc: MotorController, *, fallback_internal: bool = True) -> Optional[float]:
    """
    Try controller-report angle first; fall back to mc.get_current_angles().
    Returns mast (deg) or None if unavailable.
    """
    try:
        tup = mc._get_controller_angles()  # expected (mast, arm, ...) in deg
        if tup is not None:
            return float(tup[0])
    except Exception:
        pass
    if fallback_internal:
        try:
            mast, _ = mc.get_current_angles()
            return float(mast)
        except Exception:
            return None
    return None


def goto_abs(mc: MotorController, target_deg: float, *, tol: float, settle_s: float) -> float:
    """
    Command absolute angle (0..360 wrap), verify within tol; single automatic retry.
    Returns readback mast angle (deg).
    Raises RuntimeError on failure.
    """
    target = float(target_deg) % 360.0

    # Issue absolute move
    if not mc.rotate_mast(target):
        time.sleep(0.3)
        if not mc.rotate_mast(target):
            raise RuntimeError(f"Motor refused move to {target:.2f}°")

    # Let mechanics settle
    time.sleep(settle_s)

    # Verify
    rb = read_mast_deg(mc)
    if rb is None:
        raise RuntimeError("Cannot read mast angle from controller.")
    if abs(ang_err(rb, target)) > tol:
        # One corrective retry
        if not mc.rotate_mast(target):
            raise RuntimeError(f"Motor refused corrective move to {target:.2f}°")
        time.sleep(settle_s)
        rb = read_mast_deg(mc)
        if rb is None or abs(ang_err(rb, target)) > tol:
            raise RuntimeError(f"Angle verify miss: wanted {target:.2f}°, got {rb:.2f}°")
    return rb


# --- MAIN ROUTINE -------------------------------------------------------------
def main():
    # 1) Connect motor (no motion yet)
    mc = MotorController(MOTOR_PORT, MOTOR_BAUD)
    if not mc.connect():
        print("ERROR: Motor controller failed to connect.", file=sys.stderr)
        return 1
    print("Motor connected.")

    # 2) Connect NanoVNA and validate data path BEFORE any motion
    try:
        vna = pynanovna.VNA()  # auto-detect
    except Exception as e:
        print(f"ERROR: Failed to open NanoVNA: {e}", file=sys.stderr)
        mc.disconnect()
        return 1

    try:
        vna.set_sweep(SWEEP_START_HZ, SWEEP_STOP_HZ, SWEEP_POINTS)
        print("Validating NanoVNA sweep…")
        s11, s21, freqs = vna.sweep()  # complex arrays + frequency vector
        if not (len(freqs) and len(s11) == len(freqs) and len(s21) == len(freqs)):
            print("ERROR: NanoVNA returned no/short data—check connection/permissions.", file=sys.stderr)
            mc.disconnect()
            return 1
        print(f"VNA OK: {len(freqs)} pts {freqs[0]/1e9:.3f}→{freqs[-1]/1e9:.3f} GHz.")
    except Exception as e:
        print(f"ERROR: NanoVNA validation failed: {e}", file=sys.stderr)
        mc.disconnect()
        return 1

    # 3) Prepare CSV
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    f = CSV_PATH.open("w", newline="")
    writer = csv.writer(f)
    writer.writerow([
        "angle_deg", "freq_Hz",
        "S11_re", "S11_im", "S21_re", "S21_im",
        "S11_dB", "S21_dB",
    ])

    try:
        # 4) Home/zero like your RF helpers; best-effort
        try:
            mc.reset_orientation()
            time.sleep(0.25)
        except Exception:
            pass

        # Move to 0° and verify before any sweep
        rb0 = goto_abs(mc, 0.0, tol=ANGLE_TOL_DEG, settle_s=SETTLE_S)
        print(f"At start angle ≈ {rb0:.2f}°")

        # 5) Angle loop (absolute, exclude 360)
        angles = np.arange(0.0, 360.0, ANGLE_STEP_DEG, dtype=float)  # 0..355
        for target in angles:
            rb = goto_abs(mc, target, tol=ANGLE_TOL_DEG, settle_s=SETTLE_S)

            # Sweep and log
            s11, s21, freqs = vna.sweep()
            if not len(freqs):
                raise RuntimeError("VNA returned empty sweep; aborting for safety.")

            s11 = np.asarray(s11, dtype=np.complex128)
            s21 = np.asarray(s21, dtype=np.complex128)
            s11_db = db20(s11)
            s21_db = db20(s21)

            for k in range(len(freqs)):
                writer.writerow([
                    f"{rb:.1f}", int(freqs[k]),
                    f"{s11[k].real:.9e}", f"{s11[k].imag:.9e}",
                    f"{s21[k].real:.9e}", f"{s21[k].imag:.9e}",
                    f"{s11_db[k]:.6f}", f"{s21_db[k]:.6f}",
                ])
            f.flush()
            print(f"Angle {rb:6.1f}° → wrote {len(freqs)} pts")

        print(f"\nDone. CSV saved to: {CSV_PATH.resolve()}")

        # Return to 0° cleanly (no full-circle wrap)
        goto_abs(mc, 0.0, tol=ANGLE_TOL_DEG, settle_s=SETTLE_S)

    finally:
        # Cleanup
        try:
            f.close()
        except Exception:
            pass
        try:
            mc.disconnect()
        except Exception:
            pass
        try:
            vna.close()  # harmless if not supported
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
