"""Generate an HCMS-style howling dataset from your own audio.

Output layout (mirrors Howling_Corrupted_Music_Speech_HCMS/):
    <out>/Music/music_SSAAA.wav + .csv   kept clips (SS = source, AAA = AIR)
    <out>/Rejected/...                    clips failing the automatic QC
    <out>/AIRs/AIR_AAA.wav                feedback paths, already MSG-normalised (float32)
    <out>/Sources/music_in_SS.wav         clean 20 s source excerpts (before noise)
    <out>/sources.csv                     music_in_SS -> original file + offset
    <out>/qc_report.csv                   per-clip QC numbers

The thesis pruned clips by hand (3 experts). Here QC is automatic:
  * no_howl:     level at the howl bin rose < MIN_RISE_DB after the gain ramp
  * wrong_peak:  the strongest component at the end is not the predicted howl
Rejected clips are kept (in Rejected/) so you can listen and override.

Usage:
    python3 generate_dataset.py --beats ~/Desktop/Beats --out <dir> --n-sources 7 --n-airs 8
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import subprocess
import sys
import tempfile
import warnings

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import howling_sim as hs  # noqa: E402

FS = hs.FS
CLIP_S = 20
T_START, T_LEN = 8, 1
MSG_DB = 10.0
MIN_RISE_DB = 10.0


def decode_to_16k_mono(path: str, tmpdir: str) -> np.ndarray:
    """Decode mp3/wav/m4a/aif with macOS afconvert to 16 kHz mono float."""
    out = os.path.join(tmpdir, "decoded.wav")
    subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{FS}", "-c", "1", path, out],
                   check=True, capture_output=True)
    with warnings.catch_warnings():  # afconvert adds a harmless metadata chunk
        warnings.simplefilter("ignore", wavfile.WavFileWarning)
        fs, x = wavfile.read(out)
    assert fs == FS
    return x.astype(float) / 32768.0


def pick_excerpt(x: np.ndarray) -> tuple[np.ndarray, float] | None:
    n = CLIP_S * FS
    if len(x) < n:
        return None
    start = min(30 * FS, len(x) - n)
    seg = x[start:start + n]
    if np.sqrt(np.mean(seg ** 2)) < 1e-3:   # near-silent
        return None
    return seg, start / FS


def write_int16(path, y):
    wavfile.write(path, FS, (np.clip(y, -1, 32767 / 32768) * 32768).astype(np.int16))


def qc(y, freq_hz):
    """Howl-bin level rise (end vs pre-ramp) and whether the howl dominates the end."""
    n = 4096
    win = np.hanning(n)
    k = int(round(freq_hz * n / FS))

    def spec(a, b):
        frames = [np.abs(np.fft.rfft(y[i:i + n] * win))
                  for i in range(int(a * FS), int(b * FS) - n, n // 2)]
        return np.mean(frames, axis=0)

    pre, end = spec(2, 7.5), spec(CLIP_S - 5, CLIP_S)
    rise = 20 * np.log10(end[k - 2:k + 3].max() / (pre[k - 2:k + 3].max() + 1e-12))
    peak_hz = np.argmax(end[3:]) * FS / n + 3 * FS / n
    return float(rise), float(peak_hz), abs(peak_hz - freq_hz) <= 3 * FS / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-sources", type=int, default=7)
    ap.add_argument("--n-airs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=238)
    ap.add_argument("--end-margin-db", type=float, default=0.5,
                    help="end gain = MSG + margin. Thesis used 0 (loop only just "
                         "stable, so many clips never howl); large margins make "
                         "several frequencies howl at once.")
    args = ap.parse_args()

    for sub in ("Music", "Rejected", "AIRs", "Sources"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Feedback paths
    airs = []
    for a in range(1, args.n_airs + 1):
        F, info = hs.normalise_air_to_msg(hs.make_synthetic_air(rng), hs.DELAY_SAMPLES, FS, MSG_DB)
        name = f"AIR_{a:03d}.wav"
        wavfile.write(os.path.join(args.out, "AIRs", name), FS, F.astype(np.float32))
        airs.append((name, F, info))
        print(f"{name}: howl/MSG frequency {info.freq_hz:7.1f} Hz")

    # Sources: bounced beats in <Beats>/<month>/mp3/
    candidates = sorted(glob.glob(os.path.join(args.beats, "*", "mp3", "*.mp3")))
    order = rng.permutation(len(candidates))
    sources = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in order:
            if len(sources) == args.n_sources:
                break
            try:
                got = pick_excerpt(decode_to_16k_mono(candidates[i], tmp))
            except subprocess.CalledProcessError:
                continue
            if got:
                sources.append((candidates[i], *got))

    start_db, end_db = MSG_DB - 6, MSG_DB + args.end_margin_db
    g = hs.db2lin(hs.gain_profile_db(CLIP_S * FS, FS, start_db, end_db, T_START, T_LEN))

    with open(os.path.join(args.out, "sources.csv"), "w", newline="") as fs_csv, \
         open(os.path.join(args.out, "qc_report.csv"), "w", newline="") as qc_csv:
        src_w = csv.writer(fs_csv)
        src_w.writerow(["source", "original_file", "offset_s"])
        qc_w = csv.writer(qc_csv)
        qc_w.writerow(["clip", "msg_freq_hz", "howl_rise_db", "end_peak_hz",
                       "loudspeaker_clip_frac", "status"])
        for s_idx, (path, seg, offset) in enumerate(sources, start=1):
            src_name = f"music_in_{s_idx:02d}.wav"
            clean, _ = hs.prepare_source(seg, rng)
            write_int16(os.path.join(args.out, "Sources", src_name), clean)
            src_w.writerow([src_name, os.path.relpath(path, args.beats), f"{offset:.2f}"])
            for a_idx, (air_name, F, info) in enumerate(airs, start=1):
                noise_rng = np.random.default_rng([args.seed, s_idx, a_idx])
                _, v = hs.prepare_source(seg, noise_rng)
                out = hs.simulate_loop(v, F, g, hs.DELAY_SAMPLES, clip=1.0)
                rise, peak_hz, peak_ok = qc(out.y, info.freq_hz)
                status = "ok" if rise >= MIN_RISE_DB and peak_ok else \
                         ("no_howl" if rise < MIN_RISE_DB else "wrong_peak")
                folder = "Music" if status == "ok" else "Rejected"
                clip = f"music_{s_idx:02d}{a_idx:03d}"
                write_int16(os.path.join(args.out, folder, clip + ".wav"), out.y)
                header, row = hs.annotation_lines(src_name, air_name, start_db, end_db, FS,
                                                  T_START, T_LEN, CLIP_S, info.freq_hz)
                with open(os.path.join(args.out, folder, clip + ".csv"), "w") as f:
                    f.write(header + "\n" + row + "\n")
                qc_w.writerow([clip, f"{info.freq_hz:.1f}", f"{rise:.1f}", f"{peak_hz:.1f}",
                               f"{np.mean(np.abs(out.u) >= 1):.3f}", status])
                print(f"{clip}  {status:10s} rise {rise:5.1f} dB  peak {peak_hz:7.1f} "
                      f"/ predicted {info.freq_hz:7.1f} Hz")


if __name__ == "__main__":
    main()
