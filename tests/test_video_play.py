"""Tests for stats.video_play — the in-browser playback transcode manager.

These exercise the real cv2 codecs available on this machine:
  * mp4v write (to fabricate a tiny annotated clip), then
  * VP8 WebM write via _transcode (the actual feature under test).

If the local cv2 build can't write an mp4v clip we can read back, the
round-trip test skips gracefully (it is available on this machine, though).
Everything else (path derivation, prepare() error handling, status() default)
runs with no codec dependency.
"""
import numpy as np
import pytest

from stats import video_play


def _write_mp4v(path, n=12, w=64, h=48, fps=20.0):
    """Write a tiny mp4v clip; return True iff it wrote a readable file."""
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        return False
    for i in range(n):
        frame = np.full((h, w, 3), (i * 7) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    if not (path.exists() and path.stat().st_size > 0):
        return False
    cap = cv2.VideoCapture(str(path))
    ok = cap.isOpened() and cap.read()[0]
    cap.release()
    return ok


def test_webm_path_for_derives_play_webm_from_annotated_mp4(tmp_path):
    src = tmp_path / "abc123_annotated.mp4"
    out = video_play.webm_path_for(src)
    assert out.name == "abc123_play.webm"
    assert out.parent == src.parent
    assert out.suffix == ".webm"


def test_webm_path_for_without_annotated_suffix(tmp_path):
    # no trailing "_annotated" on the stem -> just append "_play.webm"
    src = tmp_path / "clip.mp4"
    out = video_play.webm_path_for(src)
    assert out.name == "clip_play.webm"


def test_transcode_round_trip_produces_playable_webm(tmp_path):
    src = tmp_path / "fid_annotated.mp4"
    if not _write_mp4v(src):
        pytest.skip("cv2 mp4v write unavailable in this environment")

    dst = video_play.webm_path_for(src)
    video_play._transcode("sess-rt", str(src), dst)  # synchronous, direct call

    # job state recorded as ready
    st = video_play.status("sess-rt", str(src))
    assert st["state"] == "ready"
    assert st["pct"] == 100

    # the webm exists, is non-empty, and starts with the EBML magic bytes
    assert dst.exists()
    assert dst.stat().st_size > 0
    with open(dst, "rb") as fh:
        magic = fh.read(4)
    assert magic == b"\x1a\x45\xdf\xa3"

    # and cv2 can read frames back out of it
    import cv2
    cap = cv2.VideoCapture(str(dst))
    assert cap.isOpened()
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:  # some builds report 0; fall back to counting reads
        n = sum(1 for _ in iter(lambda: cap.read()[0], False))
    cap.release()
    assert n > 0


def test_prepare_with_none_annotated_path_returns_error_and_never_raises():
    res = video_play.prepare("sess-none", None)
    assert res["state"] == "error"
    assert res["pct"] == 0
    assert "no annotated video" in res.get("message", "")


def test_prepare_with_missing_file_returns_error(tmp_path):
    missing = tmp_path / "does_not_exist_annotated.mp4"
    res = video_play.prepare("sess-missing", str(missing))
    assert res["state"] == "error"
    assert res["pct"] == 0


def test_status_unknown_sid_with_missing_path_is_none(tmp_path):
    missing = tmp_path / "nope_annotated.mp4"
    st = video_play.status("sess-unknown-xyz", str(missing))
    assert st["state"] == "none"
    assert st["pct"] == 0


def test_status_ready_when_webm_already_cached(tmp_path):
    # a non-empty webm sibling should report ready without any job registered
    src = tmp_path / "cached_annotated.mp4"
    dst = video_play.webm_path_for(src)
    dst.write_bytes(b"\x1a\x45\xdf\xa3some-bytes")
    st = video_play.status("sess-cached", str(src))
    assert st["state"] == "ready"
    assert st["pct"] == 100
