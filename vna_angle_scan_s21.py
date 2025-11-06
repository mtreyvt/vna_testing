#!/usr/bin/env python3
"""
vna_angle_scan_s21_singlefile.py
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
from serial.tools import list_ports  # optional, not required but handy

# ------------------------------------------------------------------
# Ensure MotorController.py in the same folder is importable
# ------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from MotorController import MotorController  # your existing class
# ------------------------------------------------------------------

CFG = {
    # Serial ports (use /dev/serial/by-id/... for stability on Linux)
    "motor_port":    "/dev/ttyACM0",   # Arduino/motor controller
    "motor_baud":    115200,

    "vna_port":      "/dev/ttyACM1",   # NanoVNA/OpenVNA
    "vna_baud":      115200,
    "vna_timeout_s": 5.0,              # increase if device is slow

    # Sweep parameters
    "f_start_hz":    1_000_000_000,    # 1 GHz
    "f_stop_hz":     3_000_000_000,    # 3 GHz
    "points":        401,              # reduce if timeouts happen

    # Angle scan parameters
    # Choose mode:
    #   "wrap360_from_home": start at 0°, scan to 360°, then return to 0°
    #   "centered_360": go to −180°, scan to +180°, then return to 0°
    "scan_mode":     "centered_360",
    "mast_step_deg": 5.0,
    "settle_s":      0.4,              # allow rotor to settle before scanning

    # Output CSV
    "out_dir":       "vna_scans",
    "basename":      "AUT_scan_all_angles",
    "ohms":          50.0,
}


# -------------------------- VNA Helper --------------------------
class NanoVNAInterface:
    """
    Minimal serial helper for NanoVNA/OpenVNA-style protocol.

    Primaries used:
      sweep <start> <stop> <points>
      scan  <start> <stop> <points> 0b110   (both channels)
    Fallback (older firmware):
      scan <start> <stop> <points>
      data 0   -> S11
      data 1   -> S21

    Lines returned for scan/data are: re0 im0 re1 im1 (mask) or re im (single)
    """

    def __init__(self, port: str, baud: int = 115200, timeout_s: float = 5.0):
        self.ser = serial.Serial(port=port, baudrate=baud, timeout=timeout_s)

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass

    def _write_cmd(self, cmd: str):
        # Send CRLF to be friendly across firmwares
        self.ser.write((cmd + "\r\n").encode("ascii"))

    def _read_line(self) -> str:
        line = self.ser.readline().decode("ascii", errors="ignore")
        return line.strip()

    def _drain(self, max_reads: int = 512):
        old_timeout = self.ser.timeout
        self.ser.timeout = 0.02
        try:
            for _ in range(max_reads):
                b = self.ser.read(1)
                if not b:
                    break
                _ = self.ser.readline()
        finally:
            self.ser.timeout = old_timeout

    def configure_sweep(self, f_start_hz: int, f_stop_hz: int, points: int):
        self._drain()
        self._write_cmd(f"sweep {int(f_start_hz)} {int(f_stop_hz)} {int(points)}")
        time.sleep(0.05)
        _ = self.ser.read(self.ser.in_waiting or 1)

    def _read_n_numeric_quads(self, n, timeout_s):
        """Read n lines with at least 4 float tokens (for masked scan)."""
        out = []
        old = self.ser.timeout
        self.ser.timeout = timeout_s
        try:
            while len(out) < n:
                ln = self._read_line()
                if not ln:
                    continue
                toks = ln.split()
                if len(toks) < 4:
                    # skip prompts like 'ch>' or partial echoes
                    continue
                try:
                    v = list(map(float, toks[:4]))
                    out.append(v)
                except ValueError:
                    continue
        finally:
            self.ser.timeout = old
        return out

    def _read_n_numeric_pairs(self, n, timeout_s):
        """Read n lines with at least 2 float tokens (for data 0/1)."""
        out = []
        old = self.ser.timeout
        self.ser.timeout = timeout_s
        try:
            while len(out) < n:
                ln = self._read_line()
                if not ln:
                    continue
                toks = ln.split()
                if len(toks) < 2:
                    continue
                try:
                    v = list(map(float, toks[:2]))
                    out.append(v)
                except ValueError:
                    continue
        finally:
            self.ser.timeout = old
        return out

    def scan_s11_s21(self, f_start_hz: int, f_stop_hz: int, points: int):
        """
        Try mask scan first; fall back to scan + data 0/1 (older firmwares).
        """
        # Construct frequency axis
        df = (f_stop_hz - f_start_hz) / (points - 1)
        freqs = [f_start_hz + i * df for i in range(points)]

        # --- Attempt masked scan returning both channels per line
        try:
            self._drain()
            self._write_cmd(f"scan {int(f_start_hz)} {int(f_stop_hz)} {int(points)} 0b110")
            vals = self._read_n_numeric_quads(points, max(self.ser.timeout, 5.0))
            s11 = [complex(v[0], v[1]) for v in vals]
            s21 = [complex(v[2], v[3]) for v in vals]
            if len(s11) == points and len(s21) == points:
                return freqs, s11, s21
        except Exception:
            pass  # fall through to legacy path

        # --- Legacy path: scan, then data 0 and data 1
        self._drain()
        self._write_cmd(f"sweep {int(f_start_hz)} {int(f_stop_hz)} {int(points)}")
        time.sleep(0.05)
        _ = self.ser.read(self.ser.in_waiting or 1)

        self._drain()
        self._write_cmd(f"scan {int(f_start_hz)} {int(f_stop_hz)} {int(points)}")

        self._drain()
        self._write_cmd("data 0")
        d0 = self._read_n_numeric_pairs(points, max(self.ser.timeout, 5.0))

        self._drain()
        self._write_cmd("data 1")
        d1 = self._read_n_numeric_pairs(points, max(self.ser.timeout, 5.0))

        if len(d0) != points or len(d1) != points:
            raise RuntimeError("VNA did not return expected number of points.")

        s11 = [complex(v[0], v[1]) for v in d0]
        s21 = [complex(v[0], v[1]) for v in d1]
        return freqs, s11, s21


# -------------------------- Utilities --------------------------
def mag_db(z: complex) -> float:
    m = abs(z)
    if m <= 0.0:
        return -999.0
    return 20.0 * math.log10(m)


def phase_deg(z: complex) -> float:
    return math.degrees(math.atan2(z.imag, z.real))


def angles_inclusive(start_deg: float, end_deg: float, step_deg: float):
    """Generate inclusive angles, handling both directions robustly."""
    if step_deg <= 0:
        raise ValueError("mast_step_deg must be > 0")
    acc = start_deg
    out = []
    if end_deg >= start_deg:
        while acc <= end_deg + 1e-9:
            out.append(round(acc, 6))
            acc += step_deg
    else:
        while acc >= end_deg - 1e-9:
            out.append(round(acc, 6))
            acc -= step_deg
    return out


def scan_once_with_retries(vna, f_start, f_stop, points, retries=2):
    last_err = None
    for k in range(retries + 1):
        try:
            return vna.scan_s11_s21(f_start, f_stop, points)
        except Exception as e:
            last_err = e
            time.sleep(0.2 + 0.3 * k)
    raise last_err


# ----------------------------- Main -----------------------------
def main():
    cfg = CFG.copy()

    # Prepare output path
    out_dir = HERE / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"{cfg['basename']}_{stamp}.csv"

    # Connect motor
    motor = MotorController(cfg["motor_port"], cfg["motor_baud"])
    if not motor.connect():
        raise RuntimeError("Failed to connect to motor controller.")
    print("Motor connected.")
    motor.reset_orientation()
    print("Motor homed (0 deg).")

    # Connect VNA
    vna = NanoVNAInterface(cfg["vna_port"], cfg["vna_baud"], cfg["vna_timeout_s"])
    print("VNA connected.")
    vna.configure_sweep(cfg["f_start_hz"], cfg["f_stop_hz"], cfg["points"])

    mode = cfg["scan_mode"].lower().strip()
    step = float(cfg["mast_step_deg"])
    current = 0.0

    try:
        with open(csv_path, "w", newline="") as f:
            # Prolog comments for readability
            f.write(f"# Single-file VNA sweep log; R={cfg['ohms']} ohms\n")
            f.write(f"# start_hz={cfg['f_start_hz']}, stop_hz={cfg['f_stop_hz']}, points={cfg['points']}\n")
            f.write(f"# scan_mode={cfg['scan_mode']}, step_deg={cfg['mast_step_deg']}\n")

            writer = csv.writer(f)
            writer.writerow([
                "angle_deg", "freq_hz",
                "s11_re", "s11_im",
                "s21_re", "s21_im",
                "s11_mag_db", "s21_mag_db", "s21_phase_deg"
            ])

            if mode == "wrap360_from_home":
                start = 0.0
                end = 360.0
                angles = angles_inclusive(start, end, step)

                # Ensure we are at 0 deg
                if abs(current - start) > 1e-6:
                    delta = start - current
                    print(f"Move to start {start:.1f} deg (delta {delta:+.1f} deg)")
                    motor.rotate_mast(delta)
                    current = start
                time.sleep(cfg["settle_s"])

                for idx, ang in enumerate(angles):
                    if idx > 0:
                        step_delta = ang - current
                        print(f"-> Rotate {step_delta:+.1f} deg to {ang:.1f} deg")
                        motor.rotate_mast(step_delta)
                        current = ang
                        time.sleep(cfg["settle_s"])

                    print("   Sweep…")
                    freqs, s11, s21 = scan_once_with_retries(
                        vna, cfg["f_start_hz"], cfg["f_stop_hz"], cfg["points"], retries=2
                    )
                    for f_hz, a, b in zip(freqs, s11, s21):
                        writer.writerow([
                            f"{ang:.3f}", int(f_hz),
                            f"{a.real:.9e}", f"{a.imag:.9e}",
                            f"{b.real:.9e}", f"{b.imag:.9e}",
                            f"{mag_db(a):.6f}", f"{mag_db(b):.6f}", f"{phase_deg(b):.6f}"
                        ])

                # Return home
                if abs(current) > 1e-6:
                    print("Return to home (0 deg)…")
                    motor.rotate_mast(-current)
                    current = 0.0

            elif mode == "centered_360":
                # Go to -180, scan to +180 in steps, then return to 0
                start = -180.0
                end = +180.0
                angles = angles_inclusive(start, end, step)

                print("Move to -180.0 deg")
                motor.rotate_mast(start - current)  # 0 -> -180
                current = start
                time.sleep(cfg["settle_s"])

                for idx, ang in enumerate(angles):
                    if idx > 0:
                        print(f"-> Rotate +{step:.1f} deg to {ang:.1f} deg")
                        motor.rotate_mast(step)
                        current = ang
                        time.sleep(cfg["settle_s"])

                    print("   Sweep…")
                    freqs, s11, s21 = scan_once_with_retries(
                        vna, cfg["f_start_hz"], cfg["f_stop_hz"], cfg["points"], retries=2
                    )
                    for f_hz, a, b in zip(freqs, s11, s21):
                        writer.writerow([
                            f"{ang:.3f}", int(f_hz),
                            f"{a.real:.9e}", f"{a.imag:.9e}",
                            f"{b.real:.9e}", f"{b.imag:.9e}",
                            f"{mag_db(a):.6f}", f"{mag_db(b):.6f}", f"{phase_deg(b):.6f}"
                        ])

                # Back to home from +180
                print("Return to home (0 deg)…")
                motor.rotate_mast(-current)  # +180 -> 0
                current = 0.0

            else:
                raise ValueError("Unknown scan_mode: %s" % cfg["scan_mode"])

        print(f"All data written to: {csv_path}")

    finally:
        # Always try to restore and close hardware
        try:
            if abs(current) > 1e-6:
                motor.rotate_mast(-current)
        except Exception:
            pass
        try:
            vna.close()
        except Exception:
            pass
        try:
            motor.disconnect()
        except Exception:
            pass

    print("Done.")


if __name__ == "__main__":
    main()
