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
import matplotlib.pyplot as plt

# Import polar plotting and time gating utilities
from PolarPlot import plot_polar_patterns
from TimeGating import apply_time_gating_matrix, impulse_response, apply_time_gate

# --- USER KNOBS ---------------------------------------------------------------
MOTOR_PORT      = "/dev/ttyACM1"   # e.g. "COM5" on Windows
MOTOR_BAUD      = 115200
ANGLE_STEP_DEG  = 5.0
SETTLE_S        = 0.25
TARGET_TOL_DEG  = 0.8              # acceptable |error| to target (deg)
MOVE_TIMEOUT_S  = 20.0

CSV_PATH        = Path("nanovna_pattern_5to6GHz.csv")

# Sweep settings: centre around 5.6 GHz.  By default the NanoVNA is
# capable of measuring up to 6 GHz, so sweep over 5–6 GHz to ensure the
# target frequency is included.  If your device only covers a narrower
# band, adjust these values accordingly.
SWEEP_START_HZ  = 5_000_000_000
SWEEP_STOP_HZ   = 6_000_000_000
SWEEP_POINTS    = 101              # NanoVNA-friendly; will fallback if device complains

# Number of sweeps to average per angle.  Averaging reduces noise.
AVERAGE_SWEEPS  = 3

# --- LIVE PLOTTING CONFIG ----------------------------------------------------
# Target frequency for live polar plot (Hz) – should lie within the sweep
# range.  Here we compare measurements at 5.6 GHz.
LIVE_FREQ_HZ    = 5_600_000_000  # 5.6 GHz

# Time gate width for post‑acquisition gating (nanoseconds).  This
# controls how much of the impulse response is retained.  A wider
# window preserves more multipath energy, while a narrower window
# suppresses reflections more aggressively.
TIME_GATE_NS    = 25.0           # width of Tukey window (nanoseconds)

TUKY_ALPHA      = 0.5            # Tukey window shape parameter (0→rectangular, 1→Hann)

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

        # Data holders for live plot and gating
        s21_matrix = []        # list of complex S21 arrays per angle
        pattern_vals = []      # list of dB values at LIVE_FREQ_HZ per angle

        # Setup live polar plot
        plt.ion()
        fig_polar, ax_polar = plt.subplots(subplot_kw={'projection': 'polar'})
        ax_polar.set_rlim(-40, 0)
        ax_polar.set_thetagrids(np.arange(0, 360, 15))
        ax_polar.set_title(f"Live Pattern at {LIVE_FREQ_HZ/1e9:.1f} GHz")

        for idx, a in enumerate(angles):
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

            # Store S21 for time gating and live pattern
            s21_matrix.append(s21)
            # Find index closest to LIVE_FREQ_HZ
            idx_f = int(np.argmin(np.abs(freqs - LIVE_FREQ_HZ)))
            val_db = 20*np.log10(np.clip(np.abs(s21[idx_f]), 1e-12, None))
            pattern_vals.append(val_db)

            # Update live polar plot
            # Normalise current pattern to 0 dB
            pat_db = np.array(pattern_vals)
            pat_db = pat_db - np.max(pat_db)
            theta = np.deg2rad(np.mod(angles[:len(pat_db)], 360.0))
            # Close loop for display
            ax_polar.clear()
            ax_polar.set_rlim(-40, 0)
            ax_polar.set_thetagrids(np.arange(0, 360, 15))
            ax_polar.set_title(f"Live Pattern at {LIVE_FREQ_HZ/1e9:.1f} GHz")
            if len(theta) > 1:
                theta_c = np.concatenate([theta, theta[:1]])
                pat_c   = np.concatenate([pat_db, pat_db[:1]])
                ax_polar.plot(theta_c, pat_c, 'b-', linewidth=1.5)
            plt.pause(0.01)

            print(f"Angle {a:6.1f}° (readback {rb:6.2f}°) → avg {good} sweeps → wrote {len(freqs)} pts")

        # Return to 0° cleanly (no full-circle wrap)
        print("Returning to 0.0°…")
        goto_abs(mc, 0.0)
        print(f"\nDone. CSV saved to: {CSV_PATH.resolve()}")

        # Close live plot for final results
        plt.ioff()

        # Convert lists to arrays for gating
        s21_matrix = np.array(s21_matrix, dtype=np.complex128)
        pattern_vals = np.array(pattern_vals, dtype=float)

        # Compute original pattern in dB (peak normalized)
        orig_pat_db = pattern_vals - np.max(pattern_vals)

        # Perform time gating across all frequencies using TimeGating
        gate_width_s = TIME_GATE_NS * 1e-9
        gated_pat_db = apply_time_gating_matrix(
            s21_matrix,
            freqs,
            gate_width_s=gate_width_s,
            tukey_alpha=TUKY_ALPHA,
            denoise_wavelet=True,
        )

        # Compute difference between original and gated patterns.  The
        # gating function returns a DC‑bin normalised pattern, whereas
        # the original pattern is evaluated at LIVE_FREQ_HZ.  We
        # therefore compare the two on a dB scale.
        diff_db = orig_pat_db - gated_pat_db
        main_idx = int(np.argmax(orig_pat_db))
        null_idx = int(np.argmin(orig_pat_db))
        # Identify the largest absolute null difference across all angles
        null_diff_idx = int(np.argmax(np.abs(diff_db)))
        # Print summary of peak and null differences
        print(f"\n=== Comparison vs Time‑Gated Pattern ===")
        print(f"Main‑lobe angle: {angles[main_idx]:.1f}° | Δ(original–gated) = {diff_db[main_idx]:+.2f} dB")
        print(f"Null angle:      {angles[null_idx]:.1f}° | Δ(original–gated) = {diff_db[null_idx]:+.2f} dB")
        print(f"Largest deviation across pattern occurs at {angles[null_diff_idx]:.1f}° with Δ = {diff_db[null_diff_idx]:+.2f} dB")

        # Show impulse response and gate window for a representative angle (first angle)
        try:
            from TimeGating import tukey, impulse_response, _find_gate_center_idx
            # Choose FFT length: at least twice the number of sweep points to improve
            # time resolution.  Pad further if the sweep is short.
            N_fft = max(256, 2 * s21_matrix.shape[1])
            # Compute sampling rate from frequency spacing
            df = float(np.median(np.diff(np.sort(freqs))))
            fs = df * float(N_fft)
            # Compute impulse responses for all angles
            h_t = impulse_response(s21_matrix, N_fft)
            # Determine centre index of the direct‑path peak
            center_idx = _find_gate_center_idx(h_t, fs)
            # Convert gate width in seconds to samples
            gate_len = max(1, int(np.ceil((gate_width_s) * fs)))
            start = max(0, center_idx - gate_len // 2)
            end   = min(N_fft, start + gate_len)
            # Build gate window vector
            g_vec = np.zeros(N_fft)
            g_vec[start:end] = tukey(end - start, alpha=TUKY_ALPHA)
            # Time axis in nanoseconds
            t = np.arange(N_fft) / fs * 1e9
            # Plot impulse magnitude and gate window for the first angle
            imp_mag = 20*np.log10(np.abs(h_t[0, :]) + 1e-12)
            fig2, ax2 = plt.subplots()
            ax2.plot(t, imp_mag, label='Impulse |h(t)| dB')
            ax2.set_xlabel('Time (ns)')
            ax2.set_ylabel('Magnitude (dB)')
            ax2.grid(True)
            ax2.set_title('Impulse Response and Time Gate (first angle)')
            # Secondary axis for gate window
            ax2_twin = ax2.twinx()
            ax2_twin.plot(t, g_vec, 'r', label='Gate window')
            ax2_twin.set_ylabel('Gate amplitude')
            # Indicate gate duration on plot
            gate_ns = (end - start) / fs * 1e9
            ax2.annotate(f'Gate width ≈ {gate_ns:.2f} ns', xy=(t[start], 0.5), xycoords='data',
                         xytext=(t[start], np.max(imp_mag)), textcoords='data',
                         arrowprops=dict(arrowstyle='->', lw=1.0, color='gray'),
                         fontsize=8, color='gray')
            fig2.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Failed to plot impulse and gate: {e}")

        # Plot final patterns using PolarPlot module
        try:
            plot_polar_patterns(
                angles,
                [
                    ("Original", orig_pat_db),
                    ("Time-Gated", gated_pat_db),
                ],
                rmin=-40, rmax=0, rticks=(-40, -20, 0),
                title=f"Original vs Time-Gated at {LIVE_FREQ_HZ/1e9:.1f} GHz"
            )
        except Exception as e:
            print(f"Failed to plot polar patterns: {e}")

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
