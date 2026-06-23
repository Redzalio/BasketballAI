"""Optional: detection.pose._angle.

This is intentionally SKIPPED. The _angle() helper itself is a tiny pure-numpy
geometry function, but `import detection.pose` pulls in `torch` and
`ultralytics` at module import time (pose.py lines ~15-16). The test suite's
charter is pure, torch-free logic -- no model/GPU stack -- so we do not import
it here.

If pose.py is ever refactored to lazy-import torch (e.g. move the heavy imports
inside the functions that need a model), delete the skip and the test below
will lock in the angle math:  _angle((0,1),(0,0),(1,0)) == 90.
"""
import importlib.util

import pytest

pytestmark = pytest.mark.skip(
    reason="importing detection.pose pulls in torch/ultralytics; "
           "out of scope for the torch-free logic suite"
)


def test_angle_right_angle_is_90():
    from detection.pose import _angle  # would import torch -> skipped
    assert _angle((0, 1), (0, 0), (1, 0)) == pytest.approx(90.0)
