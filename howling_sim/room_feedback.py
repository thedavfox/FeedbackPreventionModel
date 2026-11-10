"""Live-room feedback simulator for multi-stem beats.

A gig-like scene: 1-2 stems are performed into microphones on a stage, the
other stems are plugged straight into the mixer (backing track / DI keys).
The mixer drives PA speakers in a simulated room (pyroomacoustics image-source
model, MIT licence); the mics pick the speakers back up, so feedback builds
only through the mic'd channels. This is a physical closed loop:

    mic_i(t) = e_i(t) + sum_k w_k(t) sum_j (F_kij * u_j)(t)
    u_j(t)   = clip( sum_i R_ij g_i(t-d) mic_i(t-d) + sum_l A_lj line_l(t-d) )

    e_i   performers heard by mic i (own source + bleed) + mic noise
    F_kij room impulse response speaker j -> mic i at mic position k
    w_k   cross-fade weights between positions (mic walk), sum to 1
    R     mic -> speaker send matrix, A: line -> speaker send matrix
    g_i   mic channel gain (fader), d: mixer latency, clip: speaker saturation

Approximations (documented in the per-clip CSV):
  * During a mic walk the performer moves with the mic, so the performer ->
    own-mic and bleed responses are kept from the start position.
  * Speakers and performers are point sources (speakers sub-cardioid,
    performers omni); mics are cardioid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pyroomacoustics as pra
import scipy.signal as ss
from pyroomacoustics.directivities import Cardioid, DirectionVector, SubCardioid

import howling_sim as hs

SPEED_OF_SOUND = pra.constants.get("c")
PRA_GLOBAL_DELAY = pra.constants.get("frac_delay_length") // 2   # samples pra adds to every RIR
FS = 16000


# ------------------------------------------------------------------ simulator
@dataclass
class LiveOutput:
    mic: np.ndarray   # (n_mic, n) signals arriving at the mixer's mic channels
    u: np.ndarray     # (n_spk, n) speaker feeds


def simulate_live(e, F, R, g, delay, w=None, line=None, A=None, clip=1.0) -> LiveOutput:
    """Exact block simulation of the live loop (block length = mixer latency,
    so a block never depends on its own output)."""
    e = np.atleast_2d(np.asarray(e, float))
    F = np.asarray(F, float)
    K, n_mic, n_spk, L = F.shape
    n = e.shape[1]
    w = np.ones((1, n)) if w is None else np.asarray(w, float)
    g = np.atleast_2d(np.asarray(g, float))
    B = int(delay)
    if B < 1:
        raise ValueError("delay must be >= 1 sample")
    N = 1 << int(np.ceil(np.log2(L + B - 1)))
    Fh = np.fft.rfft(F, N)                                    # (K, mic, spk, Nf)
    pad = N
    mic = np.zeros((n_mic, pad + n))
    u = np.zeros((n_spk, pad + n))
    gp = np.concatenate([np.zeros((n_mic, pad)), g], axis=1)
    line_drive = np.zeros((n_spk, pad + n))
    if line is not None and len(line):
        line_drive[:, pad:] = np.asarray(A, float).T @ np.atleast_2d(line)
    for s in range(0, n, B):
        t1 = min(s + B, n)
        a, b = pad + s, pad + t1
        drive = R.T @ (gp[:, a - B:b - B] * mic[:, a - B:b - B]) + line_drive[:, a - B:b - B]
        u[:, a:b] = np.clip(drive, -clip, clip)
        U = np.fft.rfft(u[:, b - N:b], N)
        x = np.fft.irfft(np.einsum("kisf,sf->kif", Fh, U), N)[..., N - (t1 - s):]
        mic[:, a:b] = e[:, s:t1] + np.einsum("kib,kb->ib", x, w[:, s:t1])
    return LiveOutput(mic=mic[:, pad:], u=u[:, pad:])


# --------------------------------------------------------------------- MSG
def _crossings(mu, fs, nfft, fmin, fmax):
    """All frequencies where the phase of mu crosses 0, with |mu| there
    (same interpolation rule as hs.loop_msg)."""
    f = np.arange(len(mu)) * fs / nfft
    ph, mag = np.angle(mu), np.abs(mu)
    a, b = ph[:-1], ph[1:]
    cross = ((a <= 0) & (b > 0) | (a >= 0) & (b < 0)) & (np.abs(a - b) < np.pi)
    k = np.nonzero(cross & (f[:-1] >= fmin) & (f[1:] <= fmax))[0]
    frac = a[k] / (a[k] - b[k])
    return mag[k] + frac * (mag[k + 1] - mag[k]), f[k] + frac * fs / nfft


def _eig_branches(L0):
    """Eigenvalue branches of the (n_mic x n_mic x Nf) loop matrix, tracked
    continuously over frequency (n_mic <= 2)."""
    n = L0.shape[0]
    if n == 1:
        return [L0[0, 0]]
    if n != 2:
        raise NotImplementedError("mimo_msg supports 1 or 2 mics")
    a, b, c, d = L0[0, 0], L0[0, 1], L0[1, 0], L0[1, 1]
    half_tr = (a + d) / 2
    disc = np.sqrt(half_tr ** 2 - (a * d - b * c))
    e1, e2 = half_tr + disc, half_tr - disc
    stay = np.abs(e1[1:] - e1[:-1]) + np.abs(e2[1:] - e2[:-1])
    swap = np.abs(e1[1:] - e2[:-1]) + np.abs(e2[1:] - e1[:-1])
    parity = np.concatenate([[0], np.cumsum(swap < stay) % 2]).astype(bool)
    return [np.where(parity, e2, e1), np.where(parity, e1, e2)]


def mimo_crossings(F, R, g_rel, delay, fs=FS, nfft=hs.MSG_NFFT,
                   fmin=hs.F_MIN_HZ, fmax=hs.F_MAX_HZ):
    """Every candidate howl: (gain limit dB, frequency) for each loop-phase
    crossing of each eigenvalue branch, sorted most unstable first.

    Loop matrix around the mics: L0(w) = e^{-jwd} F(w) R^T diag(g_rel).
    With common fader gain c the loop goes unstable at w when c * mu(w) = 1
    for an eigenvalue mu, i.e. where mu's phase crosses 0 (generalised Nyquist)."""
    F = np.asarray(F, float)
    n_mic, n_spk, L = F.shape
    nfft = max(nfft, 1 << int(np.ceil(np.log2(L + delay))))
    Fw = np.fft.rfft(np.concatenate([np.zeros((n_mic, n_spk, delay)), F], axis=2), nfft)
    L0 = np.einsum("isf,sj->ijf", Fw, R.T * np.asarray(g_rel, float)[None, :])
    mags, freqs = [], []
    for mu in _eig_branches(L0):
        m, f = _crossings(mu, fs, nfft, fmin, fmax)
        mags.append(m)
        freqs.append(f)
    mags, freqs = np.concatenate(mags), np.concatenate(freqs)
    keep = mags > 0              # a branch can be identically 0 (e.g. a muted send)
    mags, freqs = mags[keep], freqs[keep]
    if len(mags) == 0:
        raise ValueError("no loop-phase crossing in band")
    order = np.argsort(-mags)
    return -20 * np.log10(mags[order]), freqs[order]


def mimo_msg(F, R, g_rel, delay, fs=FS, **kw) -> hs.MsgInfo:
    """Maximum stable common mic gain (dB) and the frequency that howls first."""
    limits, freqs = mimo_crossings(F, R, g_rel, delay, fs, **kw)
    return hs.MsgInfo(msg_db=float(limits[0]), freq_hz=float(freqs[0]))


def runner_up_gap_db(F, R, g_rel, delay, fs=FS, min_sep_hz=5.0):
    """dB between the first and the next distinct frequency to go unstable.
    Overshooting the limit by less than this keeps a single howl frequency."""
    limits, freqs = mimo_crossings(F, R, g_rel, delay, fs)
    other = np.abs(freqs - freqs[0]) > min_sep_hz
    return float(limits[other][0] - limits[0]) if other.any() else np.inf


# --------------------------------------------------------------------- room
@dataclass
class Transducer:
    pos: np.ndarray
    azimuth: float      # degrees, 0 = +x (towards the audience)
    colatitude: float   # degrees, 90 = horizontal


@dataclass
class Scene:
    dims: np.ndarray
    rt60: float
    speakers: list
    mics: list
    performers: list
    omni: bool = False
    speaker_kinds: list = field(default_factory=list)
    # Speaker + mic frequency response applied to every speaker->mic path:
    # real PA speakers cannot reproduce subsonic frequencies, and without this
    # the room model lets the loop go unstable at ~20 Hz.
    band_hz: tuple | None = (80.0, 7000.0)


def room_rirs(scene: Scene, mics, fs=FS, max_order=None, max_len_s=1.5):
    """Returns F (n_mic, n_spk, L) speaker->mic and H (n_perf, n_mic, L)."""
    e_abs, order = pra.inverse_sabine(scene.rt60, scene.dims)
    room = pra.ShoeBox(scene.dims, fs=fs, materials=pra.Material(e_abs),
                       max_order=order if max_order is None else max_order)
    for spk in scene.speakers:
        dirv = None if scene.omni else SubCardioid(
            DirectionVector(spk.azimuth, spk.colatitude, degrees=True))
        room.add_source(spk.pos, directivity=dirv)
    for p in scene.performers:
        room.add_source(p)
    for m in mics:
        dirv = None if scene.omni else Cardioid(
            DirectionVector(m.azimuth, m.colatitude, degrees=True))
        room.add_microphone(m.pos, directivity=dirv)
    room.compute_rir()
    L = int(fs * min(max_len_s, max(0.3, 1.2 * scene.rt60)))

    def fit(h):
        out = np.zeros(L)
        out[:min(L, len(h))] = h[:L]
        return out

    n_spk = len(scene.speakers)
    F = np.array([[fit(room.rir[i][j]) for j in range(n_spk)] for i in range(len(mics))])
    if scene.band_hz is not None:
        hp = ss.butter(4, scene.band_hz[0], btype="high", fs=fs, output="sos")
        lp = ss.butter(2, scene.band_hz[1], btype="low", fs=fs, output="sos")
        F = ss.sosfilt(lp, ss.sosfilt(hp, F, axis=-1), axis=-1)
    H = np.array([[fit(room.rir[i][n_spk + p]) for i in range(len(mics))]
                  for p in range(len(scene.performers))])
    return F, H


def _inside(p, dims, margin=0.3):
    return np.clip(p, margin, dims - margin)


def make_scene(rng, n_mic, fast=False):
    """Random stage: performers + cardioid mics at the back, PA mains at the
    stage front facing the audience, optionally a floor monitor facing performer 0."""
    if fast:
        dims, rt60 = np.array([7.0, 5.0, 3.0]), 0.25
    else:
        dims = np.array([rng.uniform(9, 20), rng.uniform(6, 14), rng.uniform(3, 6)])
        rt60 = rng.uniform(0.3, 1.2)
    W = dims[1]
    ys = [W / 2] if n_mic == 1 else [W / 2 - rng.uniform(0.8, 2), W / 2 + rng.uniform(0.8, 2)]
    performers, mics = [], []
    for y in ys:
        p = _inside(np.array([rng.uniform(1.2, 2.5), y, rng.uniform(1.5, 1.7)]), dims)
        m = _inside(p + np.array([rng.uniform(0.05, 0.3), 0, -0.05]), dims)
        performers.append(p)
        mics.append(Transducer(m, 180 + rng.uniform(-40, 40), 90 + rng.uniform(-20, 20)))
    mains_x = rng.uniform(3.0, 4.5)
    speakers, kinds = [], []
    for frac in (0.15, 0.85):
        pos = _inside(np.array([mains_x, W * frac, rng.uniform(2.0, 2.5)]), dims)
        az = np.degrees(np.arctan2(W / 2 - pos[1], dims[0] - pos[0])) + rng.uniform(-10, 10)
        speakers.append(Transducer(pos, az, 95))
        kinds.append("main")
    if rng.random() < 0.5:
        p0 = performers[0]
        pos = _inside(p0 + np.array([rng.uniform(0.8, 1.3), rng.uniform(-0.3, 0.3), 0]), dims)
        pos[2] = 0.3
        speakers.append(Transducer(pos, 180, 60))
        kinds.append("monitor")
    band = (80.0, 7000.0) if fast else (rng.uniform(60, 100), rng.uniform(6000, 7500))
    return Scene(dims=dims, rt60=rt60, speakers=speakers, mics=mics,
                 performers=performers, speaker_kinds=kinds, band_hz=band)


def routing(scene, rng):
    """Mains get every mic; a monitor gets mic 0 and some of mic 1."""
    n_mic, n_spk = len(scene.mics), len(scene.speakers)
    R = np.zeros((n_mic, n_spk))
    for j, kind in enumerate(scene.speaker_kinds):
        if kind == "main":
            R[:, j] = 1.0
        else:
            R[0, j] = 1.0
            if n_mic > 1:
                R[1, j] = rng.uniform(0, 1)
    return R


# --------------------------------------------------------------------- stems
def select_window(tracks: dict, fs=FS, clip_s=20, active_db=-50.0):
    """Use every given track (sorted by file name) and pick the clip_s window
    that maximises the summed fraction of time each track is sounding
    (0.5 s frames above active_db); ties go to the loudest window.
    Tracks shorter than the song are zero-padded at the end (Logic exports
    start at the project start, so they are already aligned).
    Returns names, X (n_tracks, clip_s*fs), offset_s, active (bool per track)."""
    names = sorted(tracks)
    n = max(len(tracks[k]) for k in names)
    if n < clip_s * fs:
        raise ValueError(f"longest track is {n / fs:.1f}s, shorter than {clip_s}s")
    X = np.zeros((len(names), n))
    for i, k in enumerate(names):
        X[i, :len(tracks[k])] = tracks[k]
    hop = fs // 2
    n_fr = n // hop
    ms = (X[:, :n_fr * hop] ** 2).reshape(len(names), n_fr, hop).mean(axis=2)
    win = int(clip_s * fs / hop)
    c = np.concatenate([np.zeros((len(names), 1)), np.cumsum(ms, axis=1)], axis=1)
    win_ms = (c[:, win:] - c[:, :-win]) / win                     # (track, start)
    on = (ms >= hs.db2lin(active_db) ** 2).astype(float)
    co = np.concatenate([np.zeros((len(names), 1)), np.cumsum(on, axis=1)], axis=1)
    frac_on = (co[:, win:] - co[:, :-win]) / win
    score = np.round(frac_on.sum(axis=0), 6) + 1e-6 * np.log10(win_ms.sum(axis=0) + 1e-30)
    start = min(int(np.argmax(score)) * hop, n - clip_s * fs)
    Xw = X[:, start:start + clip_s * fs]
    active = np.mean(Xw ** 2, axis=1) >= hs.db2lin(active_db) ** 2
    return names, Xw, start / fs, active


# --------------------------------------------------------------------- clip
@dataclass
class Clip:
    clean: np.ndarray        # (n_stems, n) scaled clean stems (line outputs == these)
    outputs: np.ndarray      # (n_stems, n) what the mixer receives per stem
    pa: np.ndarray           # (n_spk, n) speaker feeds
    mic_stems: list          # stem indices that were mic'd (impacted)
    scenario: str
    msg: hs.MsgInfo          # gain limit and howl frequency at the instability
    onset_s: float           # predicted time the loop becomes unstable
    scene: Scene
    walk_end: list           # mic positions at the end (== start for fader)
    F: np.ndarray            # (K, n_mic, n_spk, L)
    H: np.ndarray
    knots_s: np.ndarray      # times of the K mic positions
    R: np.ndarray
    A: np.ndarray
    g_rel: np.ndarray
    gain_db: np.ndarray      # common mic fader gain over time
    delay: int
    margin_db: float = 0.0
    notes: str = ""


T_RAMP, T_RAMP_LEN = 8.0, 1.0
WALK_T0, WALK_T1, WALK_K = 6.0, 14.0, 6


def _hat_weights(n, fs, knots_s):
    t = np.arange(n) / fs
    K = len(knots_s)
    if K == 1:
        return np.ones((1, n))
    w = np.zeros((K, n))
    for k in range(K):
        y = np.zeros(K)
        y[k] = 1.0
        w[k] = np.interp(t, knots_s, y)
    return w


MIN_OVERSHOOT_DB = 0.3
MAX_ROOM_TRIES = 5


def render_clip(stems, rng, fs=FS, n_mic=None, scenario=None, fast_room=False,
                target_pa_rms=0.05, snr_db=40.0, mic_stems=None, active=None,
                _tries=0) -> Clip:
    """active: optional bool per stem; mics are only put on stems that play."""
    stems = np.asarray(stems, float)
    n_st, n = stems.shape
    pool = np.arange(n_st) if active is None else np.nonzero(active)[0]
    if len(pool) == 0:
        raise ValueError("no active stems to put on a mic")
    n_mic = int(rng.choice([1, 2])) if n_mic is None else n_mic
    n_mic = min(n_mic, len(pool))
    if mic_stems is None:
        mic_stems = sorted(int(i) for i in rng.choice(pool, n_mic, replace=False))
    requested_scenario = scenario
    retry = dict(fs=fs, n_mic=n_mic, fast_room=fast_room, target_pa_rms=target_pa_rms,
                 snr_db=snr_db, mic_stems=mic_stems, active=active, _tries=_tries + 1)
    line_idx = [i for i in range(n_st) if i not in mic_stems]
    scenario = str(rng.choice(["fader", "walk"])) if scenario is None else scenario
    scene = make_scene(rng, n_mic, fast=fast_room)
    R = routing(scene, rng)
    delay = int(rng.uniform(3e-3, 10e-3) * fs)
    g_rel = np.ones(n_mic)
    if n_mic > 1:
        g_rel[1] = hs.db2lin(rng.uniform(-3, 3))
    F0, H = room_rirs(scene, scene.mics, fs)
    # How far past the stability limit the gain ends up. At exactly the limit
    # (the thesis setting) low-frequency room modes grow too slowly to howl
    # within the clip; far past it several frequencies howl at once.
    # Overshoot is also capped below the gap to the next unstable frequency,
    # otherwise a faster-growing second mode can win and the label is wrong.
    margin_db = float(rng.uniform(0.5, 1.5))
    notes = []

    knots = np.array([0.0])
    Fk = F0[None]
    walk_end = [m.pos for m in scene.mics]
    if scenario == "walk":
        # Performer 0 walks out in front of a main speaker; mic keeps pointing back at them.
        # The mic turns from pointing at the performer to pointing at the speaker.
        main = scene.speakers[int(rng.integers(0, 2))]
        m0 = scene.mics[0]
        start = m0.pos
        end = _inside(np.array([main.pos[0] + rng.uniform(1.0, 2.0),
                                main.pos[1] + rng.uniform(-0.3, 0.3), start[2]]), scene.dims)
        to_spk = main.pos - end
        az_end = np.degrees(np.arctan2(to_spk[1], to_spk[0])) + rng.uniform(-15, 15)
        col_end = np.degrees(np.arccos(to_spk[2] / np.linalg.norm(to_spk)))
        d_az = (az_end - m0.azimuth + 180) % 360 - 180
        knots = np.linspace(WALK_T0, WALK_T1, WALK_K)
        Fs = []
        for k in range(WALK_K):
            a = k / (WALK_K - 1)
            mic = Transducer(start + (end - start) * a, m0.azimuth + a * d_az,
                             m0.colatitude + a * (col_end - m0.colatitude))
            Fs.append(room_rirs(scene, [mic] + scene.mics[1:], fs)[0])
        Fk = np.array(Fs)
        walk_end = [end] + [m.pos for m in scene.mics[1:]]
        ts = np.linspace(WALK_T0, WALK_T1, 41)
        wts = np.array([np.interp(ts, knots, np.eye(WALK_K)[k]) for k in range(WALK_K)])
        infos = [mimo_msg(np.tensordot(wts[:, i], Fk, axes=1), R, g_rel, delay, fs)
                 for i in range(len(ts))]
        msgs = np.array([x.msg_db for x in infos])
        c_db = msgs[-1] + margin_db
        unstable = np.nonzero(msgs <= c_db)[0]
        gap = runner_up_gap_db(np.tensordot(wts[:, -1], Fk, axes=1), R, g_rel, delay, fs)
        margin_db = min(margin_db, 0.9 * gap)
        c_db = msgs[-1] + margin_db
        unstable = np.nonzero(msgs <= c_db)[0]
        if msgs[0] - c_db >= 3.0 and len(unstable):
            gain_db = np.full(n, c_db)
            onset = float(ts[unstable[0]])
            msg = hs.MsgInfo(msg_db=float(msgs[-1]), freq_hz=infos[-1].freq_hz)
            notes.append(f"walk MSG {msgs[0]:.1f}->{msgs[-1]:.1f} dB; "
                         f"howl at onset {infos[unstable[0]].freq_hz:.1f} Hz")
        else:
            notes.append(f"walk left only {msgs[0] - c_db:.1f} dB of headroom at the start "
                         f"(need 3); fell back to fader")
            scenario, knots, Fk, walk_end = "fader", np.array([0.0]), F0[None], [m.pos for m in scene.mics]
    if scenario == "fader":
        msg = mimo_msg(F0, R, g_rel, delay, fs)
        margin_db = min(margin_db, 0.9 * runner_up_gap_db(F0, R, g_rel, delay, fs))
        start_db, end_db = msg.msg_db - 6.0, msg.msg_db + margin_db
        gain_db = hs.gain_profile_db(n, fs, start_db, end_db, T_RAMP, T_RAMP_LEN)
        onset = T_RAMP + T_RAMP_LEN * 6.0 / (6.0 + margin_db)

    if margin_db < MIN_OVERSHOOT_DB and _tries < MAX_ROOM_TRIES:
        # Two frequencies are nearly tied: a safe overshoot would be so small
        # the howl could not build within the clip. Draw a new room.
        return render_clip(stems, rng, scenario=requested_scenario, **retry)
    notes.insert(0, f"overshoot {margin_db:.2f} dB")
    if _tries:
        notes.append(f"room redrawn {_tries}x (near-tied howl frequencies)")
    # Performer signals at each mic (own source + bleed), mixer levels, noise.
    e = np.zeros((n_mic, n))
    own = np.zeros((n_mic, n))
    for p, si in enumerate(mic_stems):
        for i in range(n_mic):
            y = ss.fftconvolve(stems[si], H[p, i])[:n]
            e[i] += y
            if i == p:
                own[p] = y
    c0 = hs.db2lin(gain_db[0])
    k = [c0 * g_rel[p] * np.sqrt(np.mean(own[p] ** 2)) / (np.sqrt(np.mean(stems[si] ** 2)) + 1e-12)
         for p, si in enumerate(mic_stems)]
    A = np.zeros((len(line_idx), len(scene.speakers)))
    A[:, [j for j, kd in enumerate(scene.speaker_kinds) if kd == "main"]] = np.mean(k)
    u0 = c0 * R.T @ (g_rel[:, None] * e) + A.T @ stems[line_idx]
    lam = target_pa_rms / (np.sqrt(np.mean(u0[0] ** 2)) + 1e-12)
    clean = stems * lam
    e = e * lam
    e += rng.standard_normal(e.shape) * (np.sqrt(np.mean(e ** 2, axis=1, keepdims=True))
                                         * hs.db2lin(-snr_db))
    g = g_rel[:, None] * hs.db2lin(gain_db)[None, :]
    w = _hat_weights(n, fs, knots)
    out = simulate_live(e, Fk, R, g, delay, w=w, line=clean[line_idx], A=A, clip=1.0)

    outputs = clean.copy()
    for p, si in enumerate(mic_stems):
        outputs[si] = out.mic[p]
    return Clip(clean=clean, outputs=outputs, pa=out.u, mic_stems=mic_stems, scenario=scenario,
                msg=msg, onset_s=onset, scene=scene, walk_end=walk_end, F=Fk, H=H,
                knots_s=knots, R=R, A=A, g_rel=g_rel, gain_db=gain_db, delay=delay,
                margin_db=margin_db, notes="; ".join(notes))
