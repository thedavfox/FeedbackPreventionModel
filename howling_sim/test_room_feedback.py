"""Tests for the multi-stem live-room feedback simulator."""

import numpy as np
import pytest

import howling_sim as hs
import room_feedback as rf

FS = 16000


# ----------------------------------------------------------------- reference
def naive_live(e, F, w, R, g, line, A, delay, clip):
    """Sample-by-sample reference of the live loop (see rf.simulate_live)."""
    K, n_mic, n_spk, L = F.shape
    n = e.shape[1]
    mic = np.zeros((n_mic, n))
    u = np.zeros((n_spk, n))
    for t in range(n):
        if t >= delay:
            drive = R.T @ (g[:, t - delay] * mic[:, t - delay])
            drive += A.T @ line[:, t - delay]
            u[:, t] = np.clip(drive, -clip, clip)
        k = np.arange(min(L, t + 1))
        conv = (F[:, :, :, k] * u[None, None, :, t - k]).sum(axis=(2, 3))  # (K, n_mic)
        mic[:, t] = e[:, t] + (w[:, t, None] * conv).sum(axis=0)
    return mic, u


def test_simulate_live_matches_naive_two_mics_two_speakers_crossfade():
    rng = np.random.default_rng(0)
    n, L, d = 1500, 40, 7
    F = rng.standard_normal((2, 2, 2, L)) * np.exp(-np.arange(L) / 10) * 0.15
    ramp = np.clip(np.linspace(-0.2, 1.2, n), 0, 1)
    w = np.stack([1 - ramp, ramp])
    e = rng.standard_normal((2, n)) * 0.1
    line = rng.standard_normal((1, n)) * 0.1
    R = np.array([[1.0, 0.5], [1.0, 0.0]])
    A = np.array([[0.7, 0.0]])
    g = np.vstack([np.full(n, 0.9), np.linspace(0.5, 1.1, n)])
    ref_mic, ref_u = naive_live(e, F, w, R, g, line, A, d, clip=0.4)
    out = rf.simulate_live(e, F, R, g, delay=d, w=w, line=line, A=A, clip=0.4)
    np.testing.assert_allclose(out.mic, ref_mic, atol=1e-10)
    np.testing.assert_allclose(out.u, ref_u, atol=1e-10)


def test_single_mic_reduces_to_existing_simulate_loop():
    rng = np.random.default_rng(1)
    F = hs.make_synthetic_air(rng)
    v = rng.standard_normal(3 * FS) * 0.03
    g = np.full(len(v), 0.7)
    a = hs.simulate_loop(v, F, g, delay=320)
    b = rf.simulate_live(v[None], F[None, None, None], np.ones((1, 1)), g[None], delay=320)
    np.testing.assert_allclose(b.mic[0], a.y, atol=1e-10)
    np.testing.assert_allclose(b.u[0], a.u, atol=1e-10)


def test_mimo_msg_reduces_to_loop_msg_for_one_mic():
    rng = np.random.default_rng(2)
    F = hs.make_synthetic_air(rng)
    a = hs.loop_msg(F, delay=320)
    b = rf.mimo_msg(F[None, None], R=np.ones((1, 1)), g_rel=np.ones(1), delay=320)
    assert b.msg_db == pytest.approx(a.msg_db, abs=1e-6)
    assert b.freq_hz == pytest.approx(a.freq_hz, abs=1e-6)


def _level_db(sig, f0, n=4096):
    spec = np.abs(np.fft.rfft(sig[-n:] * np.hanning(n)))
    k = int(round(f0 * n / FS))
    return 20 * np.log10(spec[k - 2:k + 3].max() + 1e-12), np.argmax(spec) * FS / n


def test_mimo_msg_predicts_threshold_and_frequency_two_mics():
    rng = np.random.default_rng(3)
    F = np.stack([np.stack([hs.make_synthetic_air(rng) * 0.3 for _ in range(2)])
                  for _ in range(2)])                       # (mic, spk, L)
    R = np.array([[1.0, 1.0], [1.0, 0.3]])
    g_rel = np.array([1.0, 0.7])
    info = rf.mimo_msg(F, R, g_rel, delay=80)
    e = rng.standard_normal((2, 6 * FS)) * 0.02
    n = e.shape[1]

    def run(c_db):
        g = np.outer(g_rel, np.full(n, hs.db2lin(c_db)))
        return rf.simulate_live(e, F[None], R, g, delay=80)

    below, above = run(info.msg_db - 6), run(info.msg_db + 0.5)
    assert np.abs(below.u).max() < 1.0
    lb, _ = _level_db(below.mic[0], info.freq_hz)
    la, peak = _level_db(above.mic[0], info.freq_hz)
    assert la - lb > 20
    assert abs(peak - info.freq_hz) < 3 * FS / 4096


def test_room_direct_path_delay_matches_distance():
    scene = rf.Scene(dims=np.array([8.0, 6.0, 3.5]), rt60=0.3,
                     speakers=[rf.Transducer(np.array([5.0, 2.0, 2.0]), 0, 90)],
                     mics=[rf.Transducer(np.array([2.0, 3.0, 1.5]), 180, 90)],
                     performers=[np.array([1.8, 3.0, 1.5])],
                     omni=True, band_hz=None)
    F, H = rf.room_rirs(scene, scene.mics, fs=FS, max_order=0)
    dist = np.linalg.norm(scene.speakers[0].pos - scene.mics[0].pos)
    expected = dist / rf.SPEED_OF_SOUND * FS + rf.PRA_GLOBAL_DELAY
    assert np.argmax(np.abs(F[0, 0])) == pytest.approx(expected, abs=1)
    d_src = np.linalg.norm(scene.performers[0] - scene.mics[0].pos)
    assert np.argmax(np.abs(H[0, 0])) == pytest.approx(
        d_src / rf.SPEED_OF_SOUND * FS + rf.PRA_GLOBAL_DELAY, abs=1)


def test_select_window_keeps_every_track_and_maximises_active_ones():
    rng = np.random.default_rng(4)
    tracks = {"b.wav": rng.standard_normal(60 * FS) * 0.2,
              "a.wav": rng.standard_normal(60 * FS) * 0.05,
              "c.wav": rng.standard_normal(55 * FS) * 0.1,   # shorter export
              "d.wav": np.zeros(60 * FS)}
    tracks["a.wav"][:30 * FS] = 0.0         # only plays in the second half
    names, X, offset, active = rf.select_window(tracks, fs=FS, clip_s=20)
    assert names == ["a.wav", "b.wav", "c.wav", "d.wav"]   # every track, file order
    assert X.shape == (4, 20 * FS)
    assert offset >= 30.0                                   # a is playing
    assert offset + 20 <= 55.0 + 1e-9                       # c is still playing
    assert active.tolist() == [True, True, True, False]


def test_render_clip_any_track_count_line_stems_untouched_and_mic_stems_howl():
    rng = np.random.default_rng(5)
    n = 20 * FS
    t = np.arange(n) / FS
    stems = np.stack([np.sin(2 * np.pi * 220 * t) * 0.2 + rng.standard_normal(n) * 0.02,
                      rng.standard_normal(n) * 0.1,
                      np.zeros(n),                               # silent in this window
                      np.sin(2 * np.pi * 330 * t) * 0.1,
                      np.zeros(n),                               # silent in this window
                      rng.standard_normal(n) * 0.05])
    active = np.array([True, True, False, True, False, True])
    clip = rf.render_clip(stems, rng, fs=FS, n_mic=2, scenario="fader", fast_room=True,
                          active=active)
    line_idx = [i for i in range(6) if i not in clip.mic_stems]
    assert len(clip.mic_stems) == 2 and len(line_idx) == 4
    assert all(active[i] for i in clip.mic_stems)            # mics only on playing tracks
    for i in line_idx:
        np.testing.assert_array_equal(clip.outputs[i], clip.clean[i])
    lvl_pre, _ = _level_db(clip.outputs[clip.mic_stems[0]][:7 * FS], clip.msg.freq_hz)
    lvl_end, _ = _level_db(clip.outputs[clip.mic_stems[0]], clip.msg.freq_hz)
    assert lvl_end - lvl_pre > 20
    assert 8.0 <= clip.onset_s <= 9.0
