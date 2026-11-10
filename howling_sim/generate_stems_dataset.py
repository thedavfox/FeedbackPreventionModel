"""Generate the multi-stem live-room howling dataset (Howling_Corrupted_Stems_HCS).

Input: one folder per beat containing the tracks exported from Logic
(File > Export > All Tracks as Audio Files), e.g.
    stems_raw/<beat name>/*.wav

Every track in a beat folder is used (you choose the tracks by what you put
in the folder). A 20 s window where as many tracks as possible are playing is
cut. Each of --scenes-per-beat scenes then puts 1-2 random playing tracks on
stage mics in a random room (room_feedback.render_clip), the rest go straight
to the mixer, and feedback is simulated through the real mic/speaker loop.

Output:
    <out>/Music/<clip>/<track>.wav     what the mixer receives per track (float32),
                                       same file names as in stems_raw/<beat>/
    <out>/Music/<clip>/pa_out.wav      speaker feeds, one channel per speaker (float32)
    <out>/Music/<clip>/<clip>.csv      one row per stem with roles and labels
    <out>/Rejected/<clip>/...          clips failing QC (kept so you can listen)
    <out>/Sources/<beat>/<track>.wav   clean 20 s tracks
    <out>/RIRs/<clip>.npz              impulse responses + mixer settings (POMDP reuse)
    <out>/sources.csv, qc_report.csv

Usage:
    ../.venv/bin/python generate_stems_dataset.py --stems-root ../stems_raw \
        --out ../Howling_Corrupted_Stems_HCS --scenes-per-beat 3
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import subprocess
import sys
import tempfile
import warnings

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import room_feedback as rf  # noqa: E402

AUDIO_EXT = (".wav", ".aif", ".aiff", ".mp3", ".m4a", ".caf", ".flac")
CLIP_S = 20
MIN_RISE_DB = 10.0

CLIP_HEADER = ["Stem file", "Original track", "Playing in window", "Role", "Impacted", "Mic channel",
               "Relative mic gain dB", "MSG frequency", "Instability onset", "Scenario",
               "Overshoot dB", "Start gain", "End gain", "Mixer latency ms", "Room dims",
               "RT60", "Speaker band Hz", "Speakers", "Mic start", "Mic end",
               "Sampling frequency", "Audio length", "Notes"]


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def xyz(p):
    return ";".join(f"{v:.2f}" for v in p)


def decode(path, fs, tmpdir):
    out = os.path.join(tmpdir, "decoded.wav")
    subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEF32@{fs}", "-c", "1", path, out],
                   check=True, capture_output=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", wavfile.WavFileWarning)
        sr, x = wavfile.read(out)
    assert sr == fs
    return x.astype(float)


def write_f32(path, x, fs):
    wavfile.write(path, fs, np.asarray(x, np.float32))


def qc(y, freq_hz, onset_s, fs):
    """Howl-bin rise from before the onset to the last 5 s, and whether the
    frequency that rose most is the predicted howl (a different frequency
    rising most means another mode howled, so the label would be wrong)."""
    n = 4096
    win = np.hanning(n)
    k = int(round(freq_hz * n / fs))

    def spec(a, b):
        return np.mean([np.abs(np.fft.rfft(y[i:i + n] * win))
                        for i in range(int(a * fs), int(b * fs) - n, n // 2)], axis=0)

    pre, end = spec(2.0, max(3.0, onset_s - 0.5)), spec(CLIP_S - 5, CLIP_S)
    rise = 20 * np.log10(end[k - 2:k + 3].max() / (pre[k - 2:k + 3].max() + 1e-12))
    gain = end / (pre + 1e-9 * end.max())
    gain[:3] = 0
    gain[end < 1e-3 * end.max()] = 0      # ignore bins that are silent at the end
    peak_hz = np.argmax(gain) * fs / n
    return float(rise), float(peak_hz), abs(peak_hz - freq_hz) <= 3 * fs / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes-per-beat", type=int, default=3)
    ap.add_argument("--seed", type=int, default=238)
    ap.add_argument("--fs", type=int, default=16000)
    args = ap.parse_args()
    fs = args.fs
    for sub in ("Music", "Rejected", "Sources", "RIRs"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    beats = sorted(d for d in os.listdir(args.stems_root)
                   if os.path.isdir(os.path.join(args.stems_root, d)))
    src_f = open(os.path.join(args.out, "sources.csv"), "w", newline="")
    qc_f = open(os.path.join(args.out, "qc_report.csv"), "w", newline="")
    src_w, qc_w = csv.writer(src_f), csv.writer(qc_f)
    src_w.writerow(["beat", "folder", "stem_file", "original_track", "offset_s",
                    "rms_dbfs", "playing_in_window"])
    qc_w.writerow(["clip", "beat", "scenario", "n_mic", "msg_freq_hz", "onset_s",
                   "howl_rise_db", "max_rise_hz", "pa_clip_frac", "status", "notes"])

    for b_idx, beat in enumerate(beats):
        files = sorted(f for f in glob.glob(os.path.join(args.stems_root, beat, "*"))
                       if f.lower().endswith(AUDIO_EXT))
        with tempfile.TemporaryDirectory() as tmp:
            tracks = {os.path.basename(f): decode(f, fs, tmp) for f in files}
        try:
            names, stems, offset, active = rf.select_window(tracks, fs=fs, clip_s=CLIP_S)
        except ValueError as err:
            print(f"skip {beat!r}: {err}")
            continue
        bs = slug(beat)
        os.makedirs(os.path.join(args.out, "Sources", bs), exist_ok=True)
        out_names = [os.path.splitext(nm)[0] + ".wav" for nm in names]
        print(f"{beat}: {len(names)} tracks, window {offset:.1f}-{offset + CLIP_S:.1f}s, "
              f"silent in window: {[nm for nm, a in zip(names, active) if not a] or 'none'}")
        for name, out_name, s, a in zip(names, out_names, stems, active):
            write_f32(os.path.join(args.out, "Sources", bs, out_name), s, fs)
            src_w.writerow([beat, bs, out_name, f"{beat}/{name}", f"{offset:.2f}",
                            f"{20 * np.log10(np.sqrt(np.mean(s ** 2)) + 1e-12):.1f}",
                            "yes" if a else "no"])

        for j in range(1, args.scenes_per_beat + 1):
            rng = np.random.default_rng([args.seed, b_idx, j])
            clip = rf.render_clip(stems, rng, fs=fs, active=active)
            cname = f"{bs}_s{j}"
            first = clip.mic_stems[0]
            rise, peak_hz, peak_ok = qc(clip.outputs[first], clip.msg.freq_hz, clip.onset_s, fs)
            status = "ok" if rise >= MIN_RISE_DB and peak_ok else \
                     ("no_howl" if rise < MIN_RISE_DB else "wrong_peak")
            folder = os.path.join(args.out, "Music" if status == "ok" else "Rejected", cname)
            os.makedirs(folder, exist_ok=True)
            for k, out_name in enumerate(out_names):
                write_f32(os.path.join(folder, out_name), clip.outputs[k], fs)
            write_f32(os.path.join(folder, "pa_out.wav"), clip.pa.T, fs)

            sc = clip.scene
            speakers = "|".join(f"{kind}@{xyz(s.pos)}" for kind, s in zip(sc.speaker_kinds, sc.speakers))
            with open(os.path.join(folder, cname + ".csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(CLIP_HEADER)
                for k, out_name in enumerate(out_names):
                    is_mic = k in clip.mic_stems
                    m = clip.mic_stems.index(k) if is_mic else None
                    w.writerow([
                        out_name, f"{beat}/{names[k]}", "yes" if active[k] else "no",
                        "mic" if is_mic else "line",
                        "yes" if is_mic else "no", "" if m is None else m + 1,
                        "" if m is None else f"{20 * np.log10(clip.g_rel[m]):.2f}",
                        f"{clip.msg.freq_hz:e}" if is_mic else "",
                        f"{clip.onset_s:.2f}" if is_mic else "",
                        clip.scenario, f"{clip.margin_db:.2f}",
                        f"{clip.gain_db[0]:.2f}", f"{clip.gain_db[-1]:.2f}",
                        f"{1000 * clip.delay / fs:.2f}", xyz(sc.dims), f"{sc.rt60:.2f}",
                        f"{sc.band_hz[0]:.0f}-{sc.band_hz[1]:.0f}", speakers,
                        "" if m is None else xyz(sc.mics[m].pos),
                        "" if m is None else xyz(clip.walk_end[m]),
                        fs, CLIP_S, clip.notes])
            np.savez_compressed(
                os.path.join(args.out, "RIRs", cname + ".npz"),
                F=clip.F.astype(np.float32), H=clip.H.astype(np.float32),
                knots_s=clip.knots_s, R=clip.R, A=clip.A, g_rel=clip.g_rel,
                gain_db=clip.gain_db.astype(np.float32), delay=clip.delay, fs=fs,
                mic_stems=np.array(clip.mic_stems), msg_db=clip.msg.msg_db,
                msg_freq_hz=clip.msg.freq_hz, onset_s=clip.onset_s)
            pa_clip = float(np.mean(np.abs(clip.pa[:, -5 * fs:]) >= 1))
            qc_w.writerow([cname, beat, clip.scenario, len(clip.mic_stems),
                           f"{clip.msg.freq_hz:.1f}", f"{clip.onset_s:.2f}", f"{rise:.1f}",
                           f"{peak_hz:.1f}", f"{pa_clip:.3f}", status, clip.notes])
            print(f"{cname:30s} {status:10s} {clip.scenario:5s} mics={clip.mic_stems} "
                  f"howl {clip.msg.freq_hz:7.1f} Hz @ {clip.onset_s:4.1f}s rise {rise:5.1f} dB "
                  f"max rise at {peak_hz:7.1f} Hz", flush=True)
    src_f.close()
    qc_f.close()


if __name__ == "__main__":
    main()
