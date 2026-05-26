# Alarm System

The alarm subsystem converts a stream of per-frame fall probabilities
into stable, actionable events. It owns: smoothing, debouncing,
pre-fall warning, post-fall severity assessment, and persistence.

## Finite-state machine

[`src/alarm/detector.py`](../src/alarm/detector.py) — `AlarmDetector` /
`FallState`.

```
                ┌────────────┐
                │   NORMAL   │ ◄─────────────────┐
                └────┬───────┘                   │
   smoothed_conf ≥ warning_threshold             │ smoothed drops
                     ▼                           │ below warning
                ┌────────────┐                   │
                │  IMMINENT  │ ──────────────────┘
                └────┬───────┘
   persistence_counter ≥ persistence_frames
                     ▼
                ┌────────────────────┐  upright?  ┌──────────┐
                │ IMPACT_DETECTED    │ ─────────► │  NORMAL  │
                └────┬───────────────┘            └──────────┘
   stillness_counter ≥ stillness_frames
                     ▼
                ┌────────────────────┐  upright?
                │ FALL_CONFIRMED     │ ──────────┐
                └────────────────────┘           ▼
                                            ┌──────────┐
                                            │ RECOVERED│ ── 2 s hold ──► NORMAL
                                            └──────────┘
```

### States

| State | Numeric | Visual (skeleton colour) | Meaning |
|---|---|---|---|
| `NORMAL` | 0 | green | No fall signal. |
| `IMMINENT` | 4 | yellow | Smoothed confidence is rising but persistence gate not yet hit. **Pre-fall warning.** |
| `IMPACT_DETECTED` | 1 | orange | Persistence gate triggered; awaiting stillness confirmation. |
| `FALL_CONFIRMED` | 2 | red | Stillness gate cleared. Event written to SQLite. |
| `RECOVERED` | 3 | (unset, NORMAL renders) | Subject became upright after confirmation. Held for 2 s before returning to NORMAL. |

### Smoothing

Each raw probability is passed through an exponential moving average:

```
smoothed_t = α · raw_t + (1 − α) · smoothed_{t−1}
```

Default `α = 0.3`. Lower α = heavier smoothing = more lag.

### Gates

| Gate | Default | What it does |
|---|---|---|
| `warning_threshold` | 0.55 | Lower bound for IMMINENT. Default is `confidence_threshold − 0.20`. |
| `confidence_threshold` | 0.75 | Smoothed probability above this counts as a fall candidate. |
| `persistence_frames` | 10 | Required consecutive fall candidates before IMPACT_DETECTED. |
| `stillness_duration_seconds` | 5.0 | Required non-upright duration to promote IMPACT → CONFIRMED. |
| `cooldown_seconds` | 30 | Refractory period after FALL_CONFIRMED before a new event can fire. |

The `is_subject_upright` flag is derived from feature index 0
(`torso_inclination < 45°` in the demo). If the subject becomes
upright while in IMPACT_DETECTED, the FSM returns to NORMAL — the
fall was a false start.

### Demo mode

`--demo-mode` overrides these to tighter values for presentation:

| Parameter | Default | Demo mode |
|---|---|---|
| `confidence_threshold` | 0.75 | 0.65 |
| `warning_threshold` | 0.55 | 0.45 |
| `persistence_frames` | 10 | 5 |
| `cooldown_seconds` | 30 | 15 |
| `stillness_duration_seconds` | 5.0 | 3.0 |

Tighter gates mean faster alarms at the cost of more false positives —
appropriate for live demos, not for deployment.

## Motion monitor and severity refinement

[`src/alarm/motion_monitor.py`](../src/alarm/motion_monitor.py)

After `FALL_CONFIRMED` fires, the demo activates a `MotionMonitor`
that measures the mean per-frame L2 displacement of valid keypoints.
The output is a smoothed `motion_score` and an `is_still` flag
(rolling-mean displacement below `motion_threshold`).

### Severity classification

```python
def classify_severity(still_seconds, recovered, severe_seconds=5.0, minor_seconds=1.5):
    if recovered:
        return "MINOR" if still_seconds <= minor_seconds else "MODERATE"
    if still_seconds >= severe_seconds:
        return "SEVERE"
    return "MODERATE"
```

| Outcome | Label | Meaning |
|---|---|---|
| Subject got back up within 1.5 s | `MINOR` | Likely a stumble or near-fall. |
| Subject got back up later | `MODERATE` | Recovered but slowly — worth flagging. |
| Subject still on the floor and unmoving for ≥ 5 s | `SEVERE` | Possible loss of consciousness; highest urgency. |
| Subject still on the floor but moving | `MODERATE` | Conscious but mobility-impaired. |

The event row's `severity` column starts as `CRITICAL` (written
immediately when FALL_CONFIRMED fires) and is rewritten to one of the
above once the assessment finalises. Additional metadata
(`still_seconds`, `motion_score`, `elapsed_seconds`, `recovered`)
lands in the JSON metadata column.

### Config

```yaml
alarm:
  severity:
    enabled: true                # ON by default
    motion_threshold: 0.003
    history_frames: 15
    severe_seconds: 5.0
    minor_seconds: 1.5
    assessment_timeout_seconds: 10.0
```

Assessment ends on any of: subject recovered, stillness exceeds
`severe_seconds`, or `assessment_timeout_seconds` elapse since the
fall.

## Event logging (SQLite)

[`src/alarm/logger.py`](../src/alarm/logger.py) — `EventLogger`.

Database: `logs/events.db`. Schema:

```sql
CREATE TABLE events (
    event_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp          TEXT NOT NULL,        -- ISO-8601, UTC
    severity           TEXT NOT NULL,        -- CRITICAL → SEVERE / MODERATE / MINOR
    confidence         REAL NOT NULL,        -- peak smoothed confidence
    camera_id          TEXT DEFAULT 'camera_0',
    duration_seconds   REAL,
    video_clip_path    TEXT,                 -- always NULL in default privacy mode
    metadata           TEXT,                 -- JSON blob (frame_index, motion stats, narration, ...)
    reviewed           INTEGER DEFAULT 0,    -- human-review flag
    is_false_positive  INTEGER               -- set by operator
)
```

WAL mode is enabled so the demo loop and the webapp can read/write
concurrently. The logger exposes:

| Method | Use |
|---|---|
| `log_event(severity, confidence, ...)` | Initial write on FALL_CONFIRMED. |
| `update_severity(event_id, severity)` | Rewrites the severity after refinement. |
| `update_metadata(event_id, updates)` | Merge new keys into the JSON metadata (motion stats, VLM narration). |
| `mark_reviewed(event_id, is_false_positive)` | Operator sets the review flag (used by webapp). |
| `get_events(limit, severity=...)` | Query history. |

## Inspecting events from the CLI

```bash
python -c "from src.alarm.logger import EventLogger; from pathlib import Path; \
el = EventLogger(Path('logs/events.db')); \
[print(e['event_id'], e['severity'], e['metadata']) for e in el.get_events(limit=10)]; \
el.close()"
```

## VLM narration (optional)

[`src/alarm/vlm_narrator.py`](../src/alarm/vlm_narrator.py)

Enable with `--vlm` on the demo. When a fall is confirmed, the
narrator submits the keyframe plus a short feature history to Google's
Gemini API and stores the returned natural-language caption in the
event's metadata (`narration` field). Requires `GOOGLE_API_KEY` in
`.env` or the environment.

This is **off by default** and only used for portfolio demos. No data
is sent to any third party unless `--vlm` is passed explicitly.
