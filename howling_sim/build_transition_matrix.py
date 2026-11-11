"""Estimate the 4-state Markov transition matrix for the per-track feedback
POMDP from simulated ground truth.

For each mic'd track in the stems dataset we reconstruct the "distance to
instability" curve margin(t) = MSG(t) - applied_gain(t) in dB, straight from
the simulator's stored impulse responses and gain trajectory (RIRs/<clip>.npz)
rather than any feature the model would see at inference. margin(t) is then
binned into 4 ordered severity states and the empirical state-to-state
transition counts are accumulated across every clip and every mic'd track.

Usage:
    ../.venv/bin/python build_transition_matrix.py \
        --stems-root ../Howling_Corrupted_Stems_HCS --hop-ms 10
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import room_feedback as rf  # noqa: E402

STATE_NAMES = ["stable", "building", "near_threshold", "howling"]
# Upper margin-dB bound (exclusive) for each state, most severe last.
DEFAULT_THRESHOLDS_DB = (3.0, 1.0, 0.0)  # -> stable | building | near_threshold | howling


def margin_curve(npz_path: Path) -> tuple[np.ndarray, int]:
    """Per-sample distance-to-instability (dB) for one mic'd-track clip."""
    d = np.load(npz_path, allow_pickle=True)
    F, R, g_rel = d["F"], d["R"], d["g_rel"]
    delay, fs = int(d["delay"]), int(d["fs"])
    gain_db = d["gain_db"]
    n = len(gain_db)

    msg_at_knots = np.array([rf.mimo_msg(F[i], R, g_rel, delay, fs=fs).msg_db
                              for i in range(F.shape[0])])
    if F.shape[0] > 1:
        knots_s = d["knots_s"]
        t = np.arange(n) / fs
        msg_t = np.interp(t, knots_s, msg_at_knots)
    else:
        msg_t = np.full(n, msg_at_knots[0])

    return msg_t - gain_db, fs


def to_states(margin_db: np.ndarray, thresholds_db=DEFAULT_THRESHOLDS_DB) -> np.ndarray:
    """0=stable ... 3=howling, ordered by severity (higher margin = safer)."""
    hi_to_lo = sorted(thresholds_db, reverse=True)
    state = np.zeros_like(margin_db, dtype=int)
    for edge in hi_to_lo:
        state += (margin_db < edge).astype(int)
    return state


def mic_track_clips(stems_root: Path):
    """Yield (npz_path, frame_count) for every mic'd track found via the csvs."""
    for csv_path in sorted(stems_root.glob("Music/*/*.csv")):
        clip_dir = csv_path.parent
        clip_name = csv_path.stem
        npz_path = stems_root / "RIRs" / f"{clip_name}.npz"
        if not npz_path.exists():
            continue
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not any(row["Role"] == "mic" for row in rows):
            continue
        yield npz_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems-root", type=Path, required=True)
    ap.add_argument("--hop-ms", type=float, default=10.0)
    ap.add_argument("--thresholds-db", type=float, nargs=3, default=DEFAULT_THRESHOLDS_DB)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    k = len(STATE_NAMES)
    counts = np.zeros((k, k), dtype=np.int64)
    n_clips = 0

    for npz_path in mic_track_clips(args.stems_root):
        margin_db, fs = margin_curve(npz_path)
        hop = max(1, int(round(args.hop_ms * fs / 1000)))
        frame_margin = margin_db[::hop]
        states = to_states(frame_margin, tuple(args.thresholds_db))
        np.add.at(counts, (states[:-1], states[1:]), 1)
        n_clips += 1

    if n_clips == 0:
        raise SystemExit(f"no mic'd-track clips found under {args.stems_root}")

    # Laplace smoothing so no row is all-zero (a state that's rarely/never
    # observed shouldn't produce an undefined transition distribution).
    smoothed = counts + 1
    P = smoothed / smoothed.sum(axis=1, keepdims=True)

    print(f"{n_clips} clips, {counts.sum()} transitions")
    print("states:", STATE_NAMES)
    print("counts:\n", counts)
    print("P:\n", np.round(P, 4))

    if args.out:
        np.savez(args.out, P=P, counts=counts, states=STATE_NAMES,
                  thresholds_db=args.thresholds_db, hop_ms=args.hop_ms)
        print("saved to", args.out)


if __name__ == "__main__":
    main()
