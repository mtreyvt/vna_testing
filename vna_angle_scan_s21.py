#!/usr/bin/env python3
"""
NanoVNA + MotorController antenna pattern sweep (RadioFunctions-style, fixed motion).

- Angles: 0..355° in 5° steps (absolute setpoints) but controller receives RELATIVE X moves.
- Sweep: 1–3 GHz at each angle, logging S11/S21 complex + dB.
- Safety:
  * Validate NanoVNA path before any motion.
  * Home/reset orientation, then verify every move (query MPos if available).
  * Always rotate forward (monotonic increasing) to avoid backtracking/wrap bugs.

CSV columns:
  angle_deg, freq_Hz, S11_re, S11_im, S21_re, S21_im, S11_dB, S21_dB
"""

import csv
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pynanovna  # NanoVNA control

# ---- USER SETTINGS -----------------------------------------------------------
MOTOR_PORT      = "/dev/ttyACM0"   # e.g., "COM5" on Windows
MOTOR_BAUD      = 115200
SWEEP_START_HZ  = 1_000_000_000
SWEEP_STOP_HZ   = 3_000_000_000
SWEEP_POINTS    = 401              # 101/201/401 are typical
ANGLE_STEP_DEG  = 5.0              # 0,5,...,355 (avoid 360 which == 0)
SETTLE_S        = 0.20             # pause after each move
ANGLE_TOL_DEG   = 0.5              # verification tolerance
CSV_PATH        = Path("nanovna_pattern_1to3GHz.csv")

# ---- Your MotorController (relative, GRBL-like) ------------------------------
from MotorController import MotorController  # provided in your project


# ---- Helpers (RadioFunctions-friendly) ---------------------------------------
def db20(x):
    x = np.asarray(x)
    return 20.0 * np.log10(np.clip(np.abs(x), 1e-12, None))


def query_mast_deg(mc: MotorController) -> Optional[float]:
    """
    Read mast angle from controller ('?'=>MPos:x,y,...) or fall back to software angle.
    RadioFunctions uses both paths (controller query and internal stored angles). 
    """
    try:
        tup = mc._get_controller_angles()  # returns (mast_angle, arm_angle) as strings/nums
        if tup is not None:
            return float(tup[0])
    except Exception:
        pass
    try:
        mast, _ = mc.get_current_angles()
        return float(mast)
    except Exception:
        return None


def rel_move_to_abs_forward(mc: MotorController, target_abs_deg: float) -> bool:
    """
    Convert desired ABSOLUTE angle into a RELATIVE move that always goes forward (positive)
    modulo 360. Your controller's rotate_mast() sends a relative 'G1 X<amount>' move and
    accumulates internal angles. We avoid commanding 360 (duplicate of 0). 
    """
    cur = query_mast_deg(mc)
    if cur is None:
        # If no reading is possible, just attempt absolute-as-relative from 0 baseline
        cur = 0.0
    target = float(target_abs_deg) % 360.0
    cur    = float(cur) % 360.0
    # Forward delta: how far to move positively from current to target
    delta_fwd = (target - cur) % 360.0
    if delta_fwd == 0.0:
        return True  # already there
    # Issue RELATIVE move via rotate_mast (controller expects relative) 
    ok = mc.rotate_mast(delta_fwd)
    return bool(ok)


def goto_abs_forward(mc: MotorController, target_abs_deg: float, *, tol: float, settle_s: float) -> float:
    """
    Drive forward to absolute angle (0..360 wrap) using relative G1 moves under the hood.
    Verify within 'tol' degrees using controller query when available; retry once if needed.
    """
    target = float(target_abs_deg) % 360.0

    if not rel_move_to_abs_forward(mc, target):
        # retry once
        time.sleep(0.3)
        if not rel_move_to_abs_forward(mc, target):
            raise RuntimeError(f"Motor refused move to {target:.2f}°")

    time.sleep(settle_s)

    rb = query_mast_deg(mc)
    if rb is None:
        raise RuntimeError("Cannot read mast angle from controller.")
    # error with wrap-awareness
    err = ((rb - target + 180.0) % 360.0) - 180.0
    if abs(err) > tol:
        # corrective forward nudge (rare, due to rounding/gear play)
        if not rel_move_to_abs_forward(mc, target):
            raise RuntimeError(f"Motor refused corrective move to {target:.2f}°")
        time.sleep(settle_s)
        rb = query_mast_deg(mc)
        if rb is None:
            raise RuntimeError("Cannot read mast angle after corrective move.")
        err = ((rb - target + 180.0) % 360.0) - 180.0
        if abs(err) > tol:
            raise RuntimeError(f"Angle verify miss: wanted {target:.2f}°, got {rb:.2f}°")
    return float(rb)


# ---- Main --------------------------------------------------------------------
def main():
    # 0) Connect motor (no motion yet)
    mc = MotorController(MOTOR_PORT, MOTOR_BAUD)
    if not mc.connect():
        print("ERROR: Motor controller failed to connect.", file=sys.stderr)
        return 1
    print("Motor connected.")

    # 1) Connect NanoVNA and validate BEFORE any motion
    try:
        vna = pynanovna.VNA()  # auto-detect
        vna.set_sweep(SWEEP_START_HZ, SWEEP_STOP_HZ, SWEEP_POINTS)
        print("Validating NanoVNA sweep…")
        s11, s21, freqs = vna.sweep()
        if not (len(freqs) and len(s11) == len(freqs) and len(s21) == len(freqs)):
            print("ERROR: NanoVNA returned no/short data; check cable/perm.", file=sys.stderr)
            mc.disconnect()
            return 1
        print(f"VNA OK: {len(freqs)} pts {freqs[0]/1e9:.3f}→{freqs[-1]/1e9:.3f} GHz")
    except Exception as e:
        print(f"ERROR: NanoVNA validation failed: {e}", file=sys.stderr)
        mc.disconnect()
        return 1

    # 2) Prepare CSV
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    f = CSV_PATH.open("w", newline="")
    writer = csv.writer(f)
    writer.writerow(["angle_deg","freq_Hz","S11_re","S11_im","S21_re","S21_im","S11_dB","S21_dB"])

    try:
        # 3) Home / reset orientation like RadioFunctions.InitMotor does
        try:
            mc.reset_orientation()  # homes + zeroes internal angles
            time.sleep(0.25)
        except Exception:
            pass

        # Move to 0° and verify
        rb0 = goto_abs_forward(mc, 0.0, tol=ANGLE_TOL_DEG, settle_s=SETTLE_S)
        print(f"At start angle ≈ {rb0:.2f}°")

        # 4) Absolute target list: 0, 5, ..., 355 (never 360)
        angles = np.arange(0.0, 360.0, ANGLE_STEP_DEG, dtype=float)

        # 5) Main sweep
        for target in angles:
            rb = goto_abs_forward(mc, target, tol=ANGLE_TOL_DEG, settle_s=SETTLE_S)

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

        # 6) Return to 0° cleanly (no “full wrap”)
        goto_abs_forward(mc, 0.0, tol=ANGLE_TOL_DEG, settle_s=SETTLE_S)

    finally:
        # Cleanup
        try: f.close()
        except Exception: pass
        try: mc.disconnect()
        except Exception: pass
        try: vna.close()
        except Exception: pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
