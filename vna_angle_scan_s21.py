#!/usr/bin/env python3
"""
vna_angle_scan_s21_singlefile.py  (ASCII-safe)
Rotate with your MotorController and, at each angle, trigger a 1–3 GHz sweep
on a NanoVNA/OpenVNA-like device. Append all rows into ONE CSV with angle tags.

CSV columns:
angle_deg,freq_hz,s11_re,s11_im,s21_re,s21_im,s11_mag_db,s21_mag_db,s21_phase_deg
"""

import csv
import math
import sys
import time
import datetime
from pathlib import Path

import serial  # pip install pyserial

# ------------------------------------------------------------------
# Make sure MotorController.py in the same folder is importable
# ------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from MotorController import MotorController  # same file/dir as this script
# ------------------------------------------------------------------

CFG = {
    # Motor serial
    "motor_port":    "COM3",     # e.g., "/dev/ttyUSB0" on Linux
    "motor_baud":    115200,

    # VNA serial
    "vna_port":      "COM3",     # e.g., "/dev/ttyACM0" on Linux
    "vna_baud":      115200,
    "vna_timeout_s": 2.0,

    # Sweep
    "f_start_hz":    1_000_000_000,   # 1 GHz
    "f_stop_hz":     3_000_000_000,   # 3 GHz
    "points":        401,

    # Angles
    "mast_start_deg": 0.0,
    "mast_end_deg":   360.0,
    "mast_step_deg":  5.0,
    "settle_s":       0.4,

    # Output
    "out_dir":   "vna_scans",
    "basename":  "AUT_scan_all_angles",
    "ohms":      50.0,
}


class NanoVNAInterface:
    """
    Minimal serial helper for NanoVNA/OpenVNA-style protocol.

    Uses:
      sweep <start> <stop> <points>
      scan  <start> <stop> <points> 0b110

    Each returned line (for scan) is:
      ch0.re ch0.im ch1.re ch1.im
    We treat ch0=S11, ch1=S21.
    """

    def __init__(self, port: str, baud: int = 115200, timeout_s: float = 2.0):
        self.ser = serial.Serial(port=port, baudrate=baud, timeout=timeout_s)

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass

    def _write_cmd(self, cmd: str):
        if not cmd.endswith("\r"):
            cmd = cmd + "\r"
        self.ser.write(cmd.encode("ascii"))

    def _read_line(self) -> str:
        line = self.ser.readline().decode("ascii", errors="ignore")
        return line.strip()

    def _drain(self, max_reads: int = 512):
        # drain any pending data
        old_timeout = self.ser.timeout
        self.ser.timeout = 0.02
        try:
            for _ in range(max_reads):
                if not self.ser.read(1):
                    break
                _ = self.ser.readline()
        finally:
            self.ser.timeout = old_timeout

    def configure_sweep(self, f_start_hz: int, f_stop_hz: int, points: int):
        self._drain()
        self._write_cmd(f"sweep {int(f_start_hz)} {int(f_stop_hz)} {int(points)}")
        # consume any prompt/echo
        time.sleep(0.05)
        _ = self.ser.read(self.ser.in_waiting or 1)

    def scan_s11_s21(self, f_start_hz: int, f_stop_hz: int, points: int):
        self._drain()
        self._write_cmd(f"scan {int(f_start_hz)} {int(f_stop_hz)} {int(points)} 0b110")

        lines = []
        for _ in range(points):
            ln = self._read_line()
            if not ln:
                ln = self._read_line()
            if not ln:
                raise RuntimeError("Timed out waiting for VNA scan data.")
            lines.append(ln)

        s11 = []
        s21 = []
        for ln in lines:
            parts = ln.split()
            if len(parts) < 4:
                continue
            try:
                c0 = complex(float(parts[0]), float(parts[1]))
                c1 = complex(float(parts[2]), float(parts[3]))
            except ValueError:
                continue
            s11.append(c0)
            s21.append(c1)

        if len(s11) != points or len(s21) != points:
            raise RuntimeError(f"Parsed {len(s11)} points but expected {points}.")

        # build frequency vector
        df = (f_stop_hz - f_start_hz) / (points - 1)
        freqs = [f_start_hz + i * df for i in range(points)]
        return freqs, s11, s21


def mag_db(z: complex) -> float:
    m = abs(z)
    if m <= 0.0:
        return -999.0
    return 20.0 * math.log10(m)


def phase_deg(z: complex) -> float:
    return math.degrees(math.atan2(z.imag, z.real))


def main():
    cfg = CFG.copy()

    out_dir = HERE / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"{cfg['basename']}_{stamp}.csv"

    header = [
        "angle_deg", "freq_hz",
        "s11_re", "s11_im",
        "s21_re", "s21_im",
        "s11_mag_db", "s21_mag_db", "s21_phase_deg"
    ]

    # Connect motor
    motor = MotorController(cfg["motor_port"], cfg["motor_baud"])
    if not motor.connect():
        raise RuntimeError("Failed to connect to motor controller.")
    print("Motor connected.")
    motor.reset_orientation()
    print("Motor homed.")

    # Connect VNA
    vna = NanoVNAInterface(cfg["vna_port"], cfg["vna_baud"], cfg["vna_timeout_s"])
    print("VNA connected.")
    vna.configure_sweep(cfg["f_start_hz"], cfg["f_stop_hz"], cfg["points"])

    # Build angle list
    start = float(cfg["mast_start_deg"])
    end = float(cfg["mast_end_deg"])
    step = float(cfg["mast_step_deg"])
    if step == 0.0:
        raise ValueError("mast_step_deg must be non-zero.")

    n_steps = int(round((end - start) / step))
    angles = [start + i * step for i in range(n_steps)] if n_steps > 0 else [start]

    current = 0.0

    try:
        with open(csv_path, "w", newline="") as f:
            # a few comment lines at top
            f.write(f"# Single-file VNA sweep log; R={cfg['ohms']} ohms\n")
            f.write(f"# start_hz={cfg['f_start_hz']}, stop_hz={cfg['f_stop_hz']}, points={cfg['points']}\n")
            f.write(f"# angles: start={start}, end={end}, step={step}\n")

            writer = csv.writer(f)
            writer.writerow(header)

            for ang in angles:
                delta = ang - current
                print(f"Rotating to {ang:.1f} deg (delta {delta:+.1f} deg)")
                if abs(delta) > 1e-6:
                    motor.rotate_mast(delta)
                    current = ang
                time.sleep(cfg["settle_s"])

                print("Sweeping 1-3 GHz...")
                freqs, s11, s21 = vna.scan_s11_s21(
                    cfg["f_start_hz"], cfg["f_stop_hz"], cfg["points"]
                )

                for f_hz, a, b in zip(freqs, s11, s21):
                    writer.writerow([
                        f"{ang:.3f}", int(f_hz),
                        f"{a.real:.9e}", f"{a.imag:.9e}",
                        f"{b.real:.9e}", f"{b.imag:.9e}",
                        f"{mag_db(a):.6f}", f"{mag_db(b):.6f}", f"{phase_deg(b):.6f}"
                    ])
                print(f"Appended {len(freqs)} rows at {ang:.1f} deg")

        print(f"All data written to: {csv_path}")

    finally:
        # always clean up hardware
        try:
            vna.close()
        except Exception:
            pass
        try:
            # return to zero position
            if abs(current) > 1e-6:
                motor.rotate_mast(-current)
        except Exception:
            pass
        try:
            motor.disconnect()
        except Exception:
            pass

    print("Done.")


if __name__ == "__main__":
    main()
