# Live Demo

How to run [`scripts/demo.py`](../scripts/demo.py) and what every flag
does.

## Minimal commands

```bash
# Skeleton-only — webcam, no model
python scripts/demo.py

# Webcam with the best single model
python scripts/demo.py --model models/run5_stride2_aug

# Best portfolio demo — ensemble + demo mode + video file
python scripts/demo.py \
    --source samples/fall_sample.mp4 \
    --model models/urfd \
    --model2 models/run5_stride2_aug \
    --demo-mode
```

## All CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--source` | `0` (webcam) | Video file path, or integer webcam index. |
| `--model` | none | Primary model checkpoint (`.pth`) or LOSO directory. |
| `--model2` | none | Secondary model for ensemble averaging. |
| `--demo-mode` | off | Tighter alarm gates for presentations (lower threshold, faster persistence). |
| `--explain` | off | Overlay Integrated-Gradients attribution on the skeleton. |
| `--explain-every` | `3` | Recompute attribution every N frames. |
| `--vlm` | off | Generate a Gemini natural-language caption on each confirmed fall. Requires `GOOGLE_API_KEY`. |
| `--arch` | `lstm` | `lstm` (default, supports ensemble + explain) or `stgcn`. |

`--model` and `--model2` accept either:
- A LOSO directory (uses `fold_0/best_model.pth`), or
- A direct path to a `.pth` checkpoint.

The architecture is auto-detected from `run_meta.json` next to the
checkpoint if present, so you don't need to pass `--arch` explicitly
when using the shipped models.

## Controls

| Key | Action |
|---|---|
| `q` or `ESC` | Quit. |
| (no others) | The demo is non-interactive once running — no pause / fast-forward. |

Video files **loop automatically** when they reach the end.

## What the HUD shows

```
┌─────────────────────────────────────────────┐
│  [skeleton overlay coloured by FSM state]   │
│                                             │
│  FPS: 14.8           STATE: IMMINENT        │
│  P(fall): 0.62       FALLS: 1               │
│                                             │
│  Top features (with --explain):             │
│    com_acceleration  +0.84                  │
│    torso_inclination +0.61                  │
│    bbox_aspect_ratio +0.45                  │
│                                             │
│  Narration (with --vlm):                    │
│    "A person stumbles backward and falls    │
│     to the floor near the kitchen table."   │
└─────────────────────────────────────────────┘
```

| HUD line | Source |
|---|---|
| FPS | Rolling mean of the last 30 frame durations. |
| STATE | Current `FallState.name` (or `CALIBRATING N%` / `INIT (f/W)` during warmup). |
| P(fall) | Most recent raw model probability for the fall class. |
| FALLS | Number of `FALL_CONFIRMED` events this session. |
| Top features | (only with `--explain`) signed normalized attribution. |
| Narration | (only with `--vlm`, after a confirmed fall) Gemini caption. |

## Skeleton colour scheme

| FSM state | Colour | Meaning |
|---|---|---|
| `NORMAL` | green | All clear. |
| `IMMINENT` | yellow | Pre-fall warning — rising confidence. |
| `IMPACT_DETECTED` | orange | Persistence gate cleared, awaiting stillness. |
| `FALL_CONFIRMED` | red | Event written to SQLite. |
| `RECOVERED` | (renders as NORMAL) | Subject got back up. |

When `--explain` is on, the skeleton joints are blended toward red
(positive attribution — joint pushed the prediction toward "fall") or
blue (negative attribution — joint pushed away from "fall"). The base
hue still reflects the FSM state.

## Demo mode (`--demo-mode`)

Overrides the alarm parameters for presentation:

| Parameter | Default | Demo mode |
|---|---|---|
| `confidence_threshold` | 0.75 | 0.65 |
| `warning_threshold` | 0.55 | 0.45 |
| `persistence_frames` | 10 | 5 |
| `cooldown_seconds` | 30 | 15 |
| `stillness_duration_seconds` | 5.0 | 3.0 |
| Keypoint radius | 4 px | 5 px |
| Bone thickness | 2 px | 3 px |

The intent is to make the FSM transitions visually obvious on short
clips. **Not for deployment.**

## Recommended demo clips

From project memory: UP-Fall **cam2 front-view** clips give the
cleanest visuals (100 % sensitivity at the default threshold). URFD
falls are also reliable. Phone-recorded clips work but benefit from
turning on OneEuro smoothing (`pose.smoothing.enabled: true` in
config).

## Privacy

By default the demo writes only:

- Pose keypoints (not pixels) into `logs/events.db` via the event log.
- The alarm severity, confidence, and motion metadata.

It does **not** save the raw video frames. The VLM narration path
sends a single keyframe to Google's Gemini API, but only when
`--vlm` is explicitly passed.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `No model found in <path>` | LOSO directory missing `fold_0/best_model.pth`. Re-run training or check `python scripts/download_models.py`. |
| State stuck at `INIT (f/30)` | Sliding window not yet full. Wait 2 s at 15 FPS. |
| Yellow IMMINENT flickering when standing still | Camera jitter. Enable `pose.smoothing.enabled: true` or `features.calibration.enabled: true` in `config/config.yaml`. |
| `CALIBRATING N%` never advances | No person detected during warmup, or the clip is too short. Lower `features.calibration.warmup_seconds` or use a longer clip. |
| `MediaPipe model not found at ...` | Run `python scripts/download_models.py` to fetch `models/pose_landmarker_full.task`. |
| FPS drop below 10 | Try `--explain-every 6` or disable `--explain`. ST-GCN is the slowest path — switch to `--arch lstm`. |

## Web upload demo (optional)

[`scripts/webapp.py`](../scripts/webapp.py) +
[`src/webapp/`](../src/webapp/) — FastAPI server that lets a user
upload a clip via a browser, runs the same pipeline, and displays the
processed video with FSM-coloured overlays.

```bash
python scripts/webapp.py
# open http://localhost:8000
```

The webapp shares the SQLite event log with the live demo, so events
from both are queryable in one place.
