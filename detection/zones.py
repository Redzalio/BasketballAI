"""Rough shot-zone derivation.

v1 placeholder: bucket left / center / right by the ball's launch-side x
relative to the rim. Gets replaced by true court zones once models/court.pt
(trained on the basketball-courts-class dataset) is wired in.
"""


def derive_zone(rim_center, form, frame, event):
    fw = frame.shape[1]
    x = None
    if event and event.get("ball_path"):
        x = event["ball_path"][0][0]   # oldest tracked point ~ launch side
    if x is None:
        return "center"
    ref = rim_center[0] if rim_center else fw / 2
    margin = 0.12 * fw
    if x < ref - margin:
        return "left"
    if x > ref + margin:
        return "right"
    return "center"
