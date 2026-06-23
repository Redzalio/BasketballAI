"""Rough shot-zone derivation — the FALLBACK used when the court mapper can't
label a shot.

The primary, richer zone mapping lives in detection/court.py
(CourtMapper.zone), which uses the shooter's feet (from pose) plus the detected
3-point / free-throw lines to produce real zones like "left wing 3" or "paint".
That fills event["zone"]. This module is what app.py / video_processor.py fall
back to when event["zone"] is missing — i.e. no court model, no court lines
detected, or no pose/feet this shot.

All we reliably have here is the ball's launch-side x (oldest tracked point)
and the rim x, so we can only recover the HORIZONTAL LANE, not range (2 vs 3)
or the paint — those need court lines / feet, which the court mapper owns.
We still upgrade the old coarse left/center/right to the same 5-way lane scheme
the court mapper uses, so the two paths speak the same vocabulary:

    left corner / left wing / center / right wing / right corner

Future upgrade: a homography from detected court-line intersections (or a
segmentation-annotated court model) would let even this fallback place the
shot in real court coordinates — see detection/court.py for the full note.
This function never raises; on any missing signal it returns "center".
"""

# fraction of frame width, measured from the rim x — mirrors detection/court.py
CENTER_BAND = 0.11
WING_BAND = 0.27


def _lane(x, ref_x, fw):
    """5-way horizontal lane from an x position vs the rim x."""
    dx = (x - ref_x) / float(fw)             # <0 = left of rim, >0 = right
    a = abs(dx)
    if a <= CENTER_BAND:
        return "center"
    side = "left" if dx < 0 else "right"
    return f"{side} wing" if a <= WING_BAND else f"{side} corner"


def derive_zone(rim_center, form, frame, event):
    try:
        fw = frame.shape[1]
        x = None
        if event and event.get("ball_path"):
            x = event["ball_path"][0][0]     # oldest tracked point ~ launch side
        if x is None:
            return "center"
        ref_x = rim_center[0] if rim_center else fw / 2.0
        return _lane(x, ref_x, fw)
    except Exception:
        return "center"
