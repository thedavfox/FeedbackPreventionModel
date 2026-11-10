# howling_sim

Feedback (howling) dataset generators for the CS238 live-feedback POMDP project.

## Generators

| Script | Output | What it simulates |
|---|---|---|
| `generate_dataset.py` | `../Howling_Corrupted_Beats_HCB/` | Thesis recipe (Mounir 2020, Ch. 7): a finished beat through one abstract mic/speaker loop. |
| `generate_stems_dataset.py` | `../Howling_Corrupted_Stems_HCS/` | A live gig: 1-2 of your tracks on stage mics in a simulated room, the rest plugged straight into the mixer. |

Both run the real closed loop (mic, mixer gain and latency, speaker clipping, room) rather than adding a howl tone to the audio.

## Stems dataset

1. Export the tracks you want from Logic (File > Export > All Tracks as Audio Files, WAV, effects and volume/pan on, normalise off).
2. Put them in `../stems_raw/<beat name>/`; every file in the folder is used.
3. Run:

```
../.venv/bin/python generate_stems_dataset.py --stems-root ../stems_raw \
    --out ../Howling_Corrupted_Stems_HCS --scenes-per-beat 3
```

Each clip folder holds every track as the mixer received it (mic'd tracks contain feedback, room sound and bleed; direct tracks are bit-identical to the clean input), `pa_out.wav` (one channel per speaker) and a CSV with one row per track.
The howl label for a mic'd track is the CSV `MSG frequency` from `Instability onset` onward.
`RIRs/<clip>.npz` stores the impulse responses and mixer settings, so a clip's room can be reused as a POMDP environment with `room_feedback.simulate_live`.

How a clip goes unstable (`Scenario`):
- `fader`: the mic gain ramps from 6 dB under the stability limit to just over it between 8 and 9 s (as in the thesis).
- `walk`: a performer walks out in front of a PA speaker between 6 and 14 s, turning the mic toward it.

The overshoot past the limit is kept below the gap to the next unstable frequency, so only the labelled frequency can howl.
Rooms where two frequencies are nearly tied are redrawn.
Once the speakers saturate, the howl can grow harmonics; that is realistic, and the labels only cover the main howl frequency.

## Tests

```
../.venv/bin/python -m pytest -q
```

## Licences and sources

- Method: M. Mounir, "Acoustic Event Detection: Feature, Evaluation and Dataset Design", PhD thesis, KU Leuven, 2020 (all rights reserved).
  The code is written from the thesis text; no thesis code was published or copied.
- HCMS dataset (`../Howling_Corrupted_Music_Speech_HCMS/`): downloaded from the KU Leuven FTP; no licence is stated, so ask the authors before redistributing or commercial use.
- pyroomacoustics: MIT licence.
- Audio: your own beats; the rooms and impulse responses are synthetic.
