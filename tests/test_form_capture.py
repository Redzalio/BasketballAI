"""Tests for detection.form_capture -- the pure cv2/numpy drawing layer.

Unlike detection.pose, form_capture does NOT import torch/ultralytics, so it's
safe to import in this torch-free suite. These tests lock in that annotate()
draws onto a copy (same shape, something changed, never the caller's array),
tolerates missing/garbage keypoints without raising, and that save() writes a
real non-empty file.
"""
import numpy as np
import pytest

from detection import form_capture


def _shooter_keypoints():
    """A plausible standing right-handed shooter at release, within 640x480."""
    return np.array([
        [320, 70],                       # 0  nose
        [332, 62], [308, 62],            # 1,2 eyes
        [344, 66], [296, 66],            # 3,4 ears
        [360, 130], [280, 130],          # 5,6 shoulders
        [392, 96], [256, 188],           # 7,8 elbows (R high)
        [404, 52], [240, 250],           # 9,10 wrists (R up = release)
        [346, 240], [294, 240],          # 11,12 hips
        [352, 330], [288, 330],          # 13,14 knees
        [356, 420], [284, 420],          # 15,16 ankles
    ], dtype=float)


@pytest.fixture
def blank():
    return np.zeros((480, 640, 3), np.uint8)


def test_annotate_draws_and_keeps_shape(blank):
    xy = _shooter_keypoints()
    form = {"elbow_angle": 168.0, "knee_bend": 138.0, "release_angle": 52.0,
            "lean_deg": 4.0, "follow_through_deg": 171.0}

    out = form_capture.annotate(blank, xy, hand="right", form=form)

    assert isinstance(out, np.ndarray)
    assert out.dtype == np.uint8
    assert out.shape == blank.shape
    # Something was actually drawn.
    assert not np.array_equal(out, blank)
    # The caller's array must be untouched.
    assert np.array_equal(blank, np.zeros((480, 640, 3), np.uint8))


def test_annotate_handles_missing_keypoints(blank):
    xy = _shooter_keypoints()
    # Zero out several joints -> treated as "missing" and skipped.
    for i in (3, 4, 9, 10, 15, 16):
        xy[i] = (0, 0)

    out = form_capture.annotate(blank, xy, hand="left",
                                form={"elbow_angle": 150.0})

    assert isinstance(out, np.ndarray)
    assert out.shape == blank.shape
    assert out.dtype == np.uint8


def test_annotate_all_missing_returns_same_shape(blank):
    xy = np.zeros((17, 2), dtype=float)   # every joint missing
    out = form_capture.annotate(blank, xy, hand="right", form={"knee_angle": 140})
    assert isinstance(out, np.ndarray)
    assert out.shape == blank.shape


def test_annotate_no_hand_no_form(blank):
    out = form_capture.annotate(blank, _shooter_keypoints())
    assert isinstance(out, np.ndarray)
    assert out.shape == blank.shape
    assert not np.array_equal(out, blank)


@pytest.mark.parametrize("bad", [
    np.zeros((5, 2), dtype=float),        # wrong number of rows
    np.zeros((17,), dtype=float),         # wrong ndim
    np.zeros((17, 1), dtype=float),       # wrong number of cols
    "not an array",                       # not array-like at all
    None,                                 # missing entirely
    [[1, 2, 3]],                          # ragged-ish junk
])
def test_annotate_garbage_xy_returns_image(blank, bad):
    out = form_capture.annotate(blank, bad, hand="right", form={"elbow_angle": 1})
    assert isinstance(out, np.ndarray)
    assert out.shape == blank.shape       # same-shape image, no raise


def test_annotate_out_of_bounds_points_dont_crash(blank):
    xy = _shooter_keypoints()
    xy[5] = (5000, -300)      # way off screen
    xy[6] = (-10, 9999)
    out = form_capture.annotate(blank, xy, hand="right",
                                form={"elbow_angle": 160})
    assert isinstance(out, np.ndarray)
    assert out.shape == blank.shape


def test_save_writes_nonempty_file(tmp_path, blank):
    xy = _shooter_keypoints()
    dst = form_capture.save(str(tmp_path / "f.jpg"), blank, xy, hand="right",
                            form={"elbow_angle": 168.0, "knee_bend": 138.0})
    assert dst                              # truthy path on success
    out_file = tmp_path / "f.jpg"
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_save_creates_parent_dirs(tmp_path, blank):
    nested = tmp_path / "gallery" / "session1" / "shot.png"
    dst = form_capture.save(str(nested), blank, _shooter_keypoints(),
                            hand="left", form=None)
    assert dst
    assert nested.exists()
    assert nested.stat().st_size > 0


def test_save_bad_image_returns_none(tmp_path):
    dst = form_capture.save(str(tmp_path / "x.jpg"), None,
                            _shooter_keypoints(), hand="right")
    assert dst is None
