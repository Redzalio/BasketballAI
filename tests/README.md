# HoopTracker tests

Pytest suite that locks in the **pure, torch-free logic** so future changes
can't silently break the core. These tests never load a YOLO/torch model and
never touch a camera, GPU, or network.

## Run

```bash
pip install pytest          # one-time (Python 3.x)
cd C:\Users\USER\HoopTracker
python -m pytest
```

Fast: the whole suite runs in well under a second.

## What's covered

| File | Module under test | What it locks in |
|---|---|---|
| `test_shot_logic.py` | `detection/shot_logic.py` | The `ShotTracker` make/miss state machine. Synthetic descending ball arcs through a fixed rim: an arc through the rim center -> **make**; an arc to the side -> **miss**; rim-only (no ball) / ball-below-only / sub-confidence detections -> **no attempt**. Asserts `makes`, `attempts`, `fg_pct`, event payloads, and the internal `_score` line-fit gate directly. |
| `test_consistency.py` | `stats/consistency.py` | `metric_stats` (tight metric -> high sub-score, high-variance metric -> low and flagged `biggest_inconsistency`; clamped 0..100; <3 readings dropped; bool/str skipped), `consistency_score` == average of sub-scores, `makes_vs_misses` per-metric make/miss means + deltas (and `{"enough": False}` with <3 makes or <3 misses), `what_to_work_on` focus+drill, and `session_consistency` end-to-end. |
| `test_db.py` | `stats/db.py` | Full lifecycle on an **isolated temp DB**: `create_session` -> `add_shot(form=...)` -> `finalize_session` -> `get_session` (full `form` dict survives the `form_json` round-trip, incl. metrics with no dedicated column; correct `makes`/`attempts`/`fg_pct`), plus `list_sessions` ordering/limit, `all_shots`, and `delete_session`. Closes with a round-trip into the consistency engine. |
| `test_pose.py` | `detection/pose.py` | **Skipped on purpose** — importing `pose.py` pulls in torch/ultralytics, which is out of scope for this torch-free suite. Documents how to re-enable if torch is lazy-imported. |

## Safety: the real database is never touched

`stats/db.py` stores its path in the module global `DB_PATH` and reads it fresh
on every connection. The `temp_db` fixture in `conftest.py` monkeypatches that
global to a `tmp_path` file and re-runs `init_db()`, so **all** DB operations in
the tests hit a throwaway database — `data/hooptracker.db` is left alone.

(Importing `stats.db` does run `init_db()` once against the real path, but that
is only a `CREATE TABLE IF NOT EXISTS` — it writes no rows.)
