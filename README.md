# FeedbackPreventionModel

Simulated acoustic-feedback (howling) data for a live feedback-detection POMDP.

- `howling_sim/`: simulator, dataset generators and tests; see `howling_sim/README.md`.
- `Howling_Corrupted_Beats_HCB/`: full beats through an abstract mic/speaker loop (thesis recipe).
- `Howling_Corrupted_Stems_HCS/`: multi-track beats in simulated rooms, with 1-2 tracks on stage mics that feed back.

Audio (`.wav`) and impulse responses (`.npz`) are stored with Git LFS; run `git lfs install` before cloning.

## Setup

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cd howling_sim && ../.venv/bin/python -m pytest -q
```

The audio in both datasets comes from our own beats.
The KU Leuven HCMS reference dataset is not included (no licence stated); it can be downloaded from `ftp://ftp.esat.kuleuven.be/stadius/mshehata/datasets/`.
