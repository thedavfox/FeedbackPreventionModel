"""Closed-loop howling simulator.

Reimplements the dataset recipe from M. Mounir, "Acoustic Event Detection:
Feature, Evaluation and Dataset Design" (KU Leuven PhD thesis, 2020), Ch. 7
and Sec. 8.5. The original MATLAB code was never published; this is written
from the text.

Loop (thesis Fig. 7.1):
    y(t) = v(t) + x(t)              microphone = source + feedback
    u(t) = clip(g(t-d) * y(t-d))    forward path: broadband gain, 20 ms delay,
                                    hard clip at +/-1 (loudspeaker saturation)
    x    = F * u                    feedback path = acoustic impulse response

Deviations from the thesis:
  * AIRs are synthetic (make_synthetic_air) instead of taken from OpenAIR,
    so there is no third-party licence. Real AIRs can be passed in instead.
  * Phase-condition frequencies are restricted to [F_MIN_HZ, F_MAX_HZ] so a
    DC or Nyquist "howl" can never be picked.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.signal as ss

FS = 16000
DELAY_SAMPLES = 320          # 20 ms at 16 kHz
F_MIN_HZ, F_MAX_HZ = 60.0, 7800.0
MSG_NFFT = 2 ** 18           # fine grid so MSG / howl frequency are accurate


def db2lin(db):
    return 10.0 ** (np.asarray(db) / 20.0)


# --------------------------------------------------------------------------- AIR
def make_synthetic_air(rng: np.random.Generator, fs: int = FS,
                       length_s: float = 1.0) -> np.ndarray:
    """Plausible room impulse response: direct path, sparse early reflections,
    exponentially decaying noise tail, band-limited by a speaker/mic response."""
    n = int(length_s * fs)
    h = np.zeros(n)
    d0 = int(rng.uniform(1e-3, 8e-3) * fs)
    h[d0] = 1.0
    for _ in range(rng.integers(8, 20)):
        h[d0 + int(rng.uniform(1e-3, 50e-3) * fs)] += rng.choice([-1, 1]) * rng.uniform(0.15, 0.7)
    rt60 = rng.uniform(0.2, 0.9)
    t = np.arange(n) / fs
    tail_start = d0 + int(5e-3 * fs)
    tail = rng.standard_normal(n) * np.exp(-6.91 * t / rt60) * rng.uniform(0.05, 0.3)
    tail[:tail_start] = 0
    h += tail
    lo, hi = rng.uniform(80, 200), rng.uniform(5000, 7500)
    sos = ss.butter(2, [lo, hi], btype="band", fs=fs, output="sos")
    return ss.sosfilt(sos, h)


# --------------------------------------------------------------------------- MSG
@dataclass
class MsgInfo:
    msg_db: float     # maximum stable broadband gain
    freq_hz: float    # frequency where loop gain first reaches 0 dB (howl freq)


def loop_msg(F: np.ndarray, delay: int, fs: int = FS, nfft: int = MSG_NFFT,
             fmin: float = F_MIN_HZ, fmax: float = F_MAX_HZ) -> MsgInfo:
    """MSG = -20 log10 max |F(w)| over w where the loop phase is a multiple of
    2*pi (thesis eq. 8.3-8.4). Zero crossings of the wrapped loop phase are
    located with linear interpolation on an nfft-point grid."""
    loop_ir = np.concatenate([np.zeros(delay), F])
    nfft = max(nfft, 1 << int(np.ceil(np.log2(len(loop_ir)))))
    L = np.fft.rfft(loop_ir, nfft)
    f = np.arange(len(L)) * fs / nfft
    ph, mag = np.angle(L), np.abs(L)
    a, b = ph[:-1], ph[1:]
    cross = ((a <= 0) & (b > 0) | (a >= 0) & (b < 0)) & (np.abs(a - b) < np.pi)
    k = np.nonzero(cross & (f[:-1] >= fmin) & (f[1:] <= fmax))[0]
    if len(k) == 0:
        raise ValueError("no loop-phase zero crossing in the frequency band")
    frac = a[k] / (a[k] - b[k])
    m = mag[k] + frac * (mag[k + 1] - mag[k])
    i = np.argmax(m)
    return MsgInfo(msg_db=float(-20 * np.log10(m[i])),
                   freq_hz=float(f[k[i]] + frac[i] * fs / nfft))


def normalise_air_to_msg(F, delay, fs=FS, msg_db=10.0):
    """Scale F so the closed loop has the requested MSG (thesis: 10 dB)."""
    info = loop_msg(F, delay, fs)
    Fn = F * db2lin(info.msg_db - msg_db)
    return Fn, MsgInfo(msg_db=msg_db, freq_hz=info.freq_hz)


# ----------------------------------------------------------------- gain profile
def gain_profile_db(n, fs=FS, start_db=4.0, end_db=10.0, t_start=8.0, t_len=1.0):
    """Flat at start_db, linear ramp over [t_start, t_start+t_len], flat at end_db."""
    t = np.arange(n) / fs
    ramp = np.clip((t - t_start) / t_len, 0.0, 1.0)
    return start_db + ramp * (end_db - start_db)


# ----------------------------------------------------------------- simulation
@dataclass
class LoopOutput:
    y: np.ndarray   # microphone signal (what a detector sees)
    u: np.ndarray   # loudspeaker signal
    x: np.ndarray   # feedback component of y


def simulate_loop(v, F, g, delay=DELAY_SAMPLES, clip=1.0) -> LoopOutput:
    """Run the feedback loop with exact block overlap-save convolution.

    Blocks are `delay` samples long: within a block, u depends only on y from
    at least `delay` samples earlier, so the block result equals the
    sample-by-sample loop exactly."""
    v = np.asarray(v, float)
    g = np.asarray(g, float)
    n, L, B = len(v), len(F), delay
    if B < 1:
        raise ValueError("delay must be >= 1 sample")
    N = 1 << int(np.ceil(np.log2(L + B - 1)))
    Fh = np.fft.rfft(F, N)
    pad = N  # leading zeros so every segment slice is in range
    u = np.zeros(pad + n)
    y = np.zeros(pad + n)
    x = np.zeros(pad + n)
    for s in range(0, n, B):
        e = min(s + B, n)
        a, b = pad + s, pad + e
        u[a:b] = np.clip(np.r_[np.zeros(max(0, delay - s)),
                               g[max(0, s - delay):e - delay]] *
                         y[a - delay:b - delay], -clip, clip) if e > delay else 0.0
        seg = u[b - N:b]
        x[a:b] = np.fft.irfft(np.fft.rfft(seg) * Fh, N)[N - (e - s):]
        y[a:b] = v[s:e] + x[a:b]
    return LoopOutput(y=y[pad:], u=u[pad:], x=x[pad:])


# --------------------------------------------------------------------- sources
def prepare_source(s, rng, target_rms=0.03, snr_db=40.0):
    """RMS-normalise and add white noise at a fixed SNR (thesis: 40 dB)."""
    s = np.asarray(s, float)
    s = s * (target_rms / (np.sqrt(np.mean(s ** 2)) + 1e-12))
    noise_rms = target_rms * db2lin(-snr_db)
    return s, s + rng.standard_normal(len(s)) * noise_rms


# ------------------------------------------------------------------ annotation
HEADER = ("Audio filename, AIR filename, Start gain, End gain, Sampling frequency, "
          "Start transition time, Length transition time, Audio length, MSG frequency")


def annotation_lines(audio, air, start_db, end_db, fs, t_start, t_len, length, msg_freq):
    """Header and row in the exact HCMS csv format."""
    row = (f"{audio}, {air}, {float(start_db):.1f}, {float(end_db):.1f}, {int(fs)}, "
           f"{int(t_start)}, {int(t_len)}, {int(length)}, {msg_freq:e}")
    return HEADER, row


def howling_labels(n_frames, frame_times_s, msg_freq_hz, nfft, fs=FS, t_start=8.0, t_len=1.0):
    """Binary time-frequency labels (thesis eq. 8.22): the two bins either side
    of the howl frequency, from t_start + t_len/2 onward."""
    lab = np.zeros((nfft // 2 + 1, n_frames), bool)
    k = msg_freq_hz * nfft / fs
    lab[int(np.floor(k)):int(np.ceil(k)) + 1, np.asarray(frame_times_s) >= t_start + t_len / 2] = True
    return lab
