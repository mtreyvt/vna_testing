#!/usr/bin/env python3
"""
Antenna pattern sweep with NanoVNA + motor controller.
- Rotates 0..355° in 5° steps (relative moves using MotorController.rotate_mast)
- At each angle: sweep 1–3 GHz and log S11/S21 (complex + |.| in dB) to CSV.
- Safety: validates VNA returns data before any motion; validates each move completed.
Requirements:
  pip install pynanovna pyserial
Notes:
  - This logs S11 and S21 (2-port NanoVNA). S12/S22 require reversing the DUT or a second pass.
  - MotorController API comes from your uploaded file (relative moves on 'X' axis).  :contentReference[oaicite:2]{index=2}
References:
  - pynanovna docs/examples for sweep() usage. :contentReference[oaicite:3]{index=3}
"""
import time, csv, math, sys
from pathlib import Path

import numpy as np
import pynanovna                                    # NanoVNA control (sweep/stream) :contentReference[oaicite:4]{index=4}

# --- USER SETTINGS ------------------------------------------------------------
MOTOR_PORT      = "/dev/ttyACM0"   # <- change if needed (Windows e.g. "COM5")
MOTOR_BAUD      = 115200
SWEEP_START_HZ  = 1_000_000_000
SWEEP_STOP_HZ   = 3_000_000_000
SWEEP_POINTS    = 101              # 101, 201, 401…; NanoVNA supports modest point counts per segment
ANGLE_STEP_DEG  = 5.0
SETTLE_S        = 0.20             # pause after each move before measuring
CSV_PATH        = Path("nanovna_pattern_1to3GHz.csv")

# --- IMPORT YOUR MOTOR CONTROLLER --------------------------------------------
from MotorController import MotorController         # uses GRBL-like G-codes    :contentReference[oaicite:5]{index=5}

# --- SMALL HELPERS ------------------------------------------------------------
def db20(x):
    x = np.asarray(x)
    return 20.0*np.log10(np.clip(np.abs(x), 1e-12, None))

def approx_equal(a, b, tol=0.5):
    return abs(a - b) <= tol

def read_controller_mast_deg(mc: MotorController, fallback_internal=True):
    """Try controller query; if unavailable, fall back to mc.get_current_angles()."""
    try:
        tup = mc._get_controller_angles()  # controller report like "MPos:x,y,..." in degrees
        if tup is not None:
            mast = float(tup[0])
            return mast
    except Exception:
        pass
    if fallback_internal:
        mast, _ = mc.get_current_angles()
        return float(mast)
    return None

# --- MAIN ROUTINE -------------------------------------------------------------
def main():
    # 1) Connect motor (no movement yet)
    mc = MotorController(MOTOR_PORT, MOTOR_BAUD)
    if not mc.connect():
        print("ERROR: Motor controller failed to connect.")
        sys.exit(1)
    print("Motor connected.")

    # 2) Connect NanoVNA and validate data path BEFORE any motion
    vna = pynanovna.VNA()  # auto-detects in most cases  :contentReference[oaicite:6]{index=6}
    vna.set_sweep(SWEEP_START_HZ, SWEEP_STOP_HZ, SWEEP_POINTS)
    print("Validating NanoVNA sweep…")
    s11, s21, freqs = vna.sweep()  # returns complex arrays + frequency vector  :contentReference[oaicite:7]{index=7}
    if not (len(freqs) and len(s11) == len(freqs) and len(s21) == len(freqs)):
        print("ERROR: NanoVNA returned no data—check connection and permissions.")
        mc.disconnect()
        sys.exit(1)
    print(f"VNA OK: {len(freqs)} points from {freqs[0]/1e9:.3f} to {freqs[-1]/1e9:.3f} GHz.")

    # 3) Prepare CSV
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    f = CSV_PATH.open("w", newline="")
    writer = csv.writer(f)
    writer.writerow([
        "angle_deg", "freq_Hz",
        "S11_re", "S11_im", "S21_re", "S21_im",
        "S11_dB", "S21_dB"
    ])

    try:
        # 4) Zero to a known start (best-effort)
        try:
            mc.reset_orientation()   # homes + zeroes internal angles if available
        except Exception:
            pass
        start_angle = read_controller_mast_deg(mc)
        print(f"Start mast angle ≈ {start_angle:.2f}°")

        # Move to 0° (absolute) by commanding relative delta
        delta0 = 0.0 - start_angle
        if not mc.rotate_mast(delta0):
            raise RuntimeError("Motor refused initial move to 0°")
        time.sleep(SETTLE_S)
        mast_now = read_controller_mast_deg(mc)
        if mast_now is None or not approx_equal(mast_now, 0.0, tol=1.0):
            print(f"WARNING: Expected ~0°, got {mast_now}°")

        # 5) Angle loop 0..355 in 5° steps
        angles = np.arange(0.0, 360.0, ANGLE_STEP_DEG)
        for target in angles:
            # Compute relative delta from current reading
            mast_now = read_controller_mast_deg(mc)
            if mast_now is None:
                raise RuntimeError("Cannot read mast angle from controller.")
            rel = float(target - mast_now)
            if abs(rel) > 1e-3:
                if not mc.rotate_mast(rel):
                    # Retry once
                    print("Retrying motor move…")
                    time.sleep(0.3)
                    if not mc.rotate_mast(rel):
                        raise RuntimeError(f"Motor move failed at target {target}°")
                time.sleep(SETTLE_S)

            # Verify position reached (±0.5°)
            mast_now = read_controller_mast_deg(mc)
            if not approx_equal(mast_now, target, tol=0.5):
                print(f"WARNING: angle verify miss: wanted {target:.2f}°, got {mast_now:.2f}°")

            # 6) Sweep and log
            s11, s21, freqs = vna.sweep()
            if not len(freqs):
                raise RuntimeError("VNA returned empty sweep; aborting for safety.")
            s11 = np.asarray(s11, dtype=np.complex128)
            s21 = np.asarray(s21, dtype=np.complex128)
            s11_db = db20(s11)
            s21_db = db20(s21)

            for k in range(len(freqs)):
                writer.writerow([
                    f"{target:.1f}", int(freqs[k]),
                    f"{s11[k].real:.9e}", f"{s11[k].imag:.9e}",
                    f"{s21[k].real:.9e}", f"{s21[k].imag:.9e}",
                    f"{s11_db[k]:.6f}", f"{s21_db[k]:.6f}",
                ])
            f.flush()
            print(f"Angle {target:6.1f}° → wrote {len(freqs)} pts")

        print(f"\nDone. CSV saved to: {CSV_PATH.resolve()}")

    finally:
        try:
            # Graceful return to ~0° (optional)
            mast_now = read_controller_mast_deg(mc)
            if mast_now is not None:
                mc.rotate_mast(-mast_now)
                time.sleep(SETTLE_S)
        except Exception:
            pass
        try:
            f.close()
        except Exception:
            pass
        try:
            mc.disconnect()
        except Exception:
            pass
        try:
            vna.close()   # harmless if not defined by backend
        except Exception:
            pass

if __name__ == "__main__":
    main()
