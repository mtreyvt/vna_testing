#!/usr/bin/env python3
"""
NanoVNA + MotorController antenna pattern sweep (absolute-angle, RadioFunctions-style).

• Angles: 0..355° in 5° steps (absolute set-points; no 360 wrap)
• At each angle: sweep 1–3 GHz, log S11/S21 (complex + |.| in dB) -> CSV
• Safety: validate VNA first; verify each motor move using MPos ('?') polling
• Motor behavior aligned with RadioFunctions/MotorController usage

Requires:
  pip install pyserial pynanovna
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np

# --- USER KNOBS ---------------------------------------------------------------
MOTOR_PORT      = "/dev/ttyACM1"   # e.g. "COM5" on Windows
MOTOR_BAUD      = 115200
ANGLE_STEP_DEG  = 5.0
SETTLE_S        = 0.25
TARGET_TOL_DEG  = 0.8              # acceptable |error| to target (deg)
MOVE_TIMEOUT_S  = 20.0
CSV_PATH        = Path("nanovna_pattern_1to3GHz.csv")

SWEEP_START_HZ  = 1_000_000_000
SWEEP_STOP_HZ   = 3_000_000_000
SWEEP_POINTS    = 101              # NanoVNA-friendly; will fallback if device complains
AVERAGE_SWEEPS  = 3                # <-- NEW: number of sweeps to average per angle

# --- IMPORT YOUR PROJECT DRIVERS ---------------------------------------------
from MotorController import MotorController  # GRBL wrapper (G1 X..., '?', etc.)
# RadioFunctions style & expectations: absolute angles; verify with '?'

# pyNanoVNA: minimal API (auto-detects serial)
import pynanovna


# --- HELPERS ------------------------------------------------------------------
def db20(x):
    x = np.asarray(x)
    return 20.0 * np.log10(np.clip(np.abs(x), 1e-12, None))


def _ensure_absolute_mode(mc: MotorController):
    """
    Make sure GRBL is in absolute mode (G90). Your MotorController doesn't send G90/G91,
    so we set it explicitly and clear buffers.
    """
    try:
        mc.connection.write(b"G90\n")  # absolute positioning
        _ = mc.connection.readline().decode("ascii", errors="ignore")
        mc.connection.reset_input_buffer()
        mc.connection.reset_output_buffer()
    except Exception:
        pass


def _read_mpos_deg(mc: MotorController):
    """
    Read mast angle from controller using '?' status (MPos:x,y,...).
    MotorController._get_controller_angles() returns strings; convert to float.
    """
    try:
        tup = mc._get_controller_angles()  # (mast_str, arm_str) or None
        if tup is None:
            return None
        return float(tup[0])
    except Exception:
        return None


def goto_abs(mc: MotorController, target_deg: float,
            tol_deg: float = TARGET_TOL_DEG,
            settle_s: float = SETTLE_S,
            timeout_s: float = MOVE_TIMEOUT_S) -> float:
    """
    Command an absolute mast angle and verify via MPos polling.
    Assumes GRBL absolute mode (G90) and that MotorController.rotate_mast()
    sends 'G1 X{amount}' with {amount} interpreted as an ABSOLUTE X target.
    """
    target = float(target_deg) % 360.0

    # Issue the move (absolute)
    if not mc.rotate_mast(target):
        raise RuntimeError(f"Motor refused move to {target:.2f}°")

    # Poll MPos until within tolerance or timeout
    t0 = time.time()
    last_read = None
    while True:
        time.sleep(0.05)
        last_read = _read_mpos_deg(mc)
        if last_read is not None:
            # shortest signed error on circle
            err = ((last_read - target + 540.0) % 360.0) - 180.0
            if abs(err) <= tol_deg:
                time.sleep(settle_s)  # tiny extra settle
                return last_read
        if (time.time() - t0) > timeout_s:
            raise RuntimeError(
                f"Angle verify timeout: wanted {target:.2f}°, last={last_read!r}"
            )


def validate_vna_or_fallback(vna, start_hz, stop_hz, points):
    """
    Set sweep and try one read. If the device complains about points, try common fallbacks.
    """
    candidates = [points, 201, 101, 51]
    last_err = None
    for p in candidates:
        try:
            vna.set_sweep(start_hz, stop_hz, p)
            s11, s21, freqs = vna.sweep()
            if len(freqs) and len(s11) == len(freqs) and len(s21) == len(freqs):
                return s11, s21, freqs  # success on validation run
        except Exception as e:
            msg = str(e)
            # Common NanoVNA message: 'sweep points exceeds range ...'
            last_err = msg
            continue
    raise RuntimeError(f"NanoVNA validation failed: {last_err or 'no data returned'}")


# --- MAIN ---------------------------------------------------------------------
def main():
    # 1) Motor connect (no movement yet)
    mc = MotorController(MOTOR_PORT, MOTOR_BAUD)
    if not mc.connect():
        print("ERROR: Motor controller failed to connect.")
        sys.exit(1)
    print("Motor connected.")
    _ensure_absolute_mode(mc)

    # 2) NanoVNA connect + validation BEFORE motion
    vna = pynanovna.VNA()  # auto-detect
    print("Validating NanoVNA sweep…")
    try:
        s11, s21, freqs = validate_vna_or_fallback(
            vna, SWEEP_START_HZ, SWEEP_STOP_HZ, SWEEP_POINTS
        )
    except Exception as e:
        print(e)
        try:
            mc.disconnect()
        except Exception:
            pass
        sys.exit(1)
    print(f"VNA OK: {len(freqs)} pts from {freqs[0]/1e9:.3f} to {freqs[-1]/1e9:.3f} GHz.")

    # 3) CSV setup
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    f = CSV_PATH.open("w", newline="")
    writer = csv.writer(f)
    writer.writerow([
        "angle_deg", "freq_Hz",
        "S11_re", "S11_im", "S21_re", "S21_im",
        "S11_dB", "S21_dB"
    ])

    try:
        # Home to 0 ° explicitly (absolute) and verify
        print("Homing to 0.0°…")
        goto_abs(mc, 0.0)

        # Build absolute angle list exactly like RadioFunctions scans (no 360 wrap)
        angles = np.arange(0.0, 360.0, ANGLE_STEP_DEG)  # 0,5,...,355
        print(f"Scanning {len(angles)} angles…")

        for a in angles:
            # Absolute move + verify
            rb = goto_abs(mc, float(a))
            # small settle
            time.sleep(SETTLE_S)

            # --------- NEW: multiple sweeps averaged (complex) ----------
            s11_acc = None
            s21_acc = None
            freqs_ref = None
            good = 0
            for _ in range(int(max(1, AVERAGE_SWEEPS))):
                s11_i, s21_i, freqs_i = vna.sweep()
                if not len(freqs_i):
                    continue
                # Initialize accumulators on first good sweep
                if s11_acc is None:
                    s11_acc = np.asarray(s11_i, dtype=np.complex128)
                    s21_acc = np.asarray(s21_i, dtype=np.complex128)
                    freqs_ref = np.asarray(freqs_i, dtype=int)
                    good = 1
                else:
                    # Require consistent length; skip if mismatch
                    if len(freqs_i) != len(freqs_ref):
                        continue
                    s11_acc += np.asarray(s11_i, dtype=np.complex128)
                    s21_acc += np.asarray(s21_i, dtype=np.complex128)
                    good += 1

            if good == 0:
                raise RuntimeError("VNA returned empty sweep(s); aborting for safety.")

            s11 = s11_acc / good
            s21 = s21_acc / good
            freqs = freqs_ref

            s11_db = db20(s11)
            s21_db = db20(s21)
            # ------------------------------------------------------------

            # Log each frequency row (averaged results)
            for k in range(len(freqs)):
                writer.writerow([
                    f"{a:.1f}", int(freqs[k]),
                    f"{s11[k].real:.9e}", f"{s11[k].imag:.9e}",
                    f"{s21[k].real:.9e}", f"{s21[k].imag:.9e}",
                    f"{s11_db[k]:.6f}", f"{s21_db[k]:.6f}",
                ])
            f.flush()
            print(f"Angle {a:6.1f}° (readback {rb:6.2f}°) → avg {good} sweeps → wrote {len(freqs)} pts")

        # Return to 0° cleanly (no full-circle wrap)
        print("Returning to 0.0°…")
        goto_abs(mc, 0.0)
        print(f"\nDone. CSV saved to: {CSV_PATH.resolve()}")

    finally:
        # Try to leave things tidy
        try:
            f.close()
        except Exception:
            pass
        try:
            vna.close()
        except Exception:
            pass
        try:
            mc.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
