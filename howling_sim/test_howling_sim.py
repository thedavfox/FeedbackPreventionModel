"""Tests for the closed-loop howling simulator (thesis Ch. 7 / Sec. 8.5 recipe)."""

import numpy as np
import pytest

import howling_sim as hs

FS = 16000


def naive_loop(v, F, g, delay, clip):
    """Sample-by-sample reference implementation of the feedback loop."""
    n = len(v)
    u = np.zeros(n)
    y = np.zeros(n)
    for t in range(n):
        if t >= delay:
            u[t] = np.clip(g[t - delay] * y[t - delay], -clip, clip)
        k = np.arange(min(len(F), t + 1))
        x_t = np.dot(F[k], u[t - k])
        y[t] = v[t] + x_t
    return y, u


def test_block_simulation_matches_naive_loop():
    rng = np.random.default_rng(0)
    F = rng.standard_normal(60) * np.exp(-np.arange(60) / 15) * 0.2
    v = rng.standard_normal(2000) * 0.1
    g = np.full(2000, 0.8)
    y_ref, u_ref = naive_loop(v, F, g, delay=8, clip=1.0)
    out = hs.simulate_loop(v, F, g, delay=8, clip=1.0)
    np.testing.assert_allclose(out.y, y_ref, atol=1e-10)
    np.testing.assert_allclose(out.u, u_ref, atol=1e-10)


def test_air_is_normalised_to_target_msg():
    rng = np.random.default_rng(1)
    F = hs.make_synthetic_air(rng, fs=FS)
    assert len(F) == FS
    Fn, info = hs.normalise_air_to_msg(F, delay=hs.DELAY_SAMPLES, fs=FS, msg_db=10.0)
    again = hs.loop_msg(Fn, delay=hs.DELAY_SAMPLES, fs=FS)
    assert again.msg_db == pytest.approx(10.0, abs=1e-6)
    assert again.freq_hz == pytest.approx(info.freq_hz, abs=0.5)
    assert hs.F_MIN_HZ <= info.freq_hz <= hs.F_MAX_HZ


def test_gain_profile_matches_thesis():
    g_db = hs.gain_profile_db(n=20 * FS, fs=FS, start_db=4.0, end_db=10.0,
                              t_start=8.0, t_len=1.0)
    assert g_db[0] == 4.0 and g_db[int(7.99 * FS)] == 4.0
    assert g_db[int(8.5 * FS)] == pytest.approx(7.0, abs=1e-3)
    assert g_db[int(9.0 * FS)] == 10.0 and g_db[-1] == 10.0


def _band_db(sig, f0, fs, n=4096):
    seg = sig[-n:] * np.hanning(n)
    spec = np.abs(np.fft.rfft(seg))
    k = int(round(f0 * n / fs))
    return 20 * np.log10(spec[k - 2:k + 3].max() + 1e-12), np.argmax(spec) * fs / n


def test_stable_below_msg_and_howls_at_predicted_frequency_above():
    rng = np.random.default_rng(2)
    F, info = hs.normalise_air_to_msg(hs.make_synthetic_air(rng, fs=FS),
                                      delay=hs.DELAY_SAMPLES, fs=FS, msg_db=10.0)
    v = rng.standard_normal(6 * FS) * 0.03

    below = hs.simulate_loop(v, F, np.full(len(v), hs.db2lin(4.0)),
                             delay=hs.DELAY_SAMPLES, clip=1.0)
    assert np.abs(below.u).max() < 1.0  # never saturates when 6 dB below MSG

    # Just above MSG only the predicted frequency is unstable. Far above MSG
    # (e.g. +3 dB) many frequencies are unstable and another one can win.
    above = hs.simulate_loop(v, F, np.full(len(v), hs.db2lin(10.5)),
                             delay=hs.DELAY_SAMPLES, clip=1.0)
    lvl_below, _ = _band_db(below.y, info.freq_hz, FS)
    lvl_above, peak_hz = _band_db(above.y, info.freq_hz, FS)
    assert lvl_above - lvl_below > 20
    assert abs(peak_hz - info.freq_hz) < 3 * FS / 4096


def test_csv_row_matches_hcms_format():
    header, row = hs.annotation_lines(
        audio="music_in_02.wav", air="AIR_010.wav", start_db=4.0, end_db=10.0,
        fs=16000, t_start=8, t_len=1, length=20, msg_freq=251.3088)
    assert header == ("Audio filename, AIR filename, Start gain, End gain, "
                      "Sampling frequency, Start transition time, "
                      "Length transition time, Audio length, MSG frequency")
    assert row == ("music_in_02.wav, AIR_010.wav, 4.0, 10.0, 16000, 8, 1, 20, "
                   "2.513088e+02")
