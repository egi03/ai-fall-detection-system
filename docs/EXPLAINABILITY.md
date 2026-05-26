# Explainability

Two complementary tools for understanding model decisions: a
gradient-based attribution overlay (always available) and an
optional LLM narration (Gemini).

## Integrated Gradients overlay

[`src/explainability/attribution.py`](../src/explainability/attribution.py)
+ [`src/explainability/joint_mapping.py`](../src/explainability/joint_mapping.py)

### What it does

Computes Integrated Gradients of the BiLSTM's "fall" output with
respect to the 15-feature input over the current sliding window, then
projects feature-level attribution back onto skeleton joints. Joints
that pushed the model toward "fall" are tinted red; joints that pushed
it away are tinted blue. The base skeleton colour still reflects the
FSM state.

### Enable

```bash
python scripts/demo.py --model models/run5_stride2_aug --explain
```

Optional: `--explain-every N` (default 3) recomputes attribution every
N frames. Lower numbers are smoother but slower; the default is a good
balance.

### HUD addition

When `--explain` is on, the bottom of the HUD lists the three highest
absolute-attribution features for the current window, with a
normalized score in [-1, 1]:

```
  com_acceleration   +0.84
  torso_inclination  +0.61
  hip_shoulder_angle -0.42
```

### Where the math lives

```python
explainer = IntegratedGradients(model, target_class=1, n_steps=8)
result    = explainer.explain(sequence)        # sequence shape (30, 15)
result.feature_attribution                     # (15,) signed scores
joint_scores = joints_from_feature_attribution(result.feature_attribution)
```

`n_steps=8` is enough to stabilize the ranking on this 15-feature
input. Each call is sub-millisecond on CPU.

### Limitations

- Works for the BiLSTM (`--arch lstm`) only. ST-GCN attribution
  requires a different mapping and is not implemented.
- Attribution is local (this window only). To explain a *decision*
  rather than an *output*, average across the window where the FSM
  fired.
- The mapping from features to joints is a documented heuristic in
  `joint_mapping.py`, not a learned model. It correctly attributes
  e.g. `left_knee_angle` to the left knee joint, but blended features
  like `mean_visibility` distribute uniformly.

## VLM narration (optional)

[`src/alarm/vlm_narrator.py`](../src/alarm/vlm_narrator.py)

### What it does

On every `FALL_CONFIRMED`, the narrator submits the keyframe plus the
last 60 frames of feature history to Google's Gemini API and stores
the returned natural-language caption in the event row's metadata
JSON, under the `narration` key. The caption also appears in the HUD
for ~6 seconds.

### Enable

1. Create a `.env` file at the repo root:
   ```
   GOOGLE_API_KEY=ya29...
   ```
2. Run the demo with `--vlm`:
   ```bash
   python scripts/demo.py --source clip.mp4 --model models/run5_stride2_aug --vlm
   ```

The narrator runs in a background thread; the live FPS is unaffected
because the API call completes asynchronously. When the caption
arrives, the database row is updated via `EventLogger.update_metadata`.

### Privacy

Only one frame is sent per confirmed fall, and only with the user's
own API key. Nothing is sent unless `--vlm` is explicitly passed.

### Example caption

```
"A person walking through a kitchen suddenly trips on a chair and
falls forward onto the floor. After landing, they remain still."
```

The caption is informational only — it does not influence the alarm
decision, which has already fired by the time the call is made.

## Inspecting attribution + narration after a session

```bash
python -c "from src.alarm.logger import EventLogger; \
from pathlib import Path; import json; \
el = EventLogger(Path('logs/events.db')); \
[print(e['event_id'], e['severity'], json.loads(e['metadata'])) \
 for e in el.get_events(limit=5)]; \
el.close()"
```

For events written by the demo with `--vlm` enabled, the metadata
will include `narration`. For events written with `--explain` enabled,
top-attribution features are not currently persisted to SQLite —
they live only in the HUD. (Adding them to the metadata is a
candidate enhancement.)

## Other explainability tools in the repo

- [`scripts/attention_visualization.py`](../scripts/attention_visualization.py) —
  ST-GCN attention heatmaps for a chosen sequence.
- [`scripts/regenerate_ablation_plot.py`](../scripts/regenerate_ablation_plot.py) —
  Visualizes the feature-ablation results as a bar chart.
- `DAILY LOGS/15.3/FEATURE_CORRELATION.md` — correlation clusters
  across the 15 features.
- `DAILY LOGS/15.3/ACTIVITY_CONFUSION.md` — per-activity confusion
  patterns from the UP-Fall evaluation.

These are batch / offline tools rather than live overlays.
