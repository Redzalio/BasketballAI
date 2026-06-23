"""Manual shot-correction + hard-example capture (Part A/B)."""
from stats import db
from stats import hard_examples


def test_flip_updates_result_and_recomputes(temp_db):
    db.init_db()
    sid = db.create_session("video", "clip.mp4")
    db.add_shot(sid, "make")
    miss_id = db.add_shot(sid, "miss")
    s = db.recompute_session(sid)
    assert s["makes"] == 1 and s["attempts"] == 2

    rsid = db.update_shot_result(miss_id, "make")
    assert rsid == sid
    s = db.recompute_session(sid)
    assert s["makes"] == 2 and s["attempts"] == 2 and s["fg_pct"] == 100.0

    sh = db.get_shot(miss_id)
    assert sh["result"] == "make" and sh["made"] == 1 and sh["manual"] == 1


def test_delete_phantom_shot(temp_db):
    db.init_db()
    sid = db.create_session("video", "c.mp4")
    db.add_shot(sid, "make")
    phantom = db.add_shot(sid, "make")
    assert db.recompute_session(sid)["attempts"] == 2

    assert db.delete_shot(phantom) == sid
    s = db.recompute_session(sid)
    assert s["attempts"] == 1 and s["makes"] == 1
    assert db.get_shot(phantom) is None


def test_add_manual_shot(temp_db):
    db.init_db()
    sid = db.create_session("live", "cam 0")
    nid = db.add_manual_shot(sid, "make", t=1.5, zone="paint")
    assert nid
    sh = db.get_shot(nid)
    assert sh["result"] == "make" and sh["made"] == 1 and sh["manual"] == 1 and sh["zone"] == "paint"
    assert db.recompute_session(sid) == {"makes": 1, "attempts": 1, "fg_pct": 100.0}


def test_corrections_log(temp_db):
    db.init_db()
    assert db.corrections_count() == 0
    sid = db.create_session("video", "c.mp4")
    s = db.add_shot(sid, "miss")
    db.log_correction(sid, s, "flip", "miss", "make", 2.0)
    assert db.corrections_count() == 1


def test_missing_ids_return_none(temp_db):
    db.init_db()
    assert db.update_shot_result(999999, "make") is None
    assert db.delete_shot(999999) is None
    assert db.get_shot(999999) is None


def test_capture_no_video_is_safe(temp_db):
    db.init_db()
    sid = db.create_session("live", "cam 0")       # live -> no video_path
    sess = db.get_session(sid)["session"]
    assert hard_examples.capture_for_correction(sess, 1, 1.0, "flip", "miss", "make") is None
    m = hard_examples.manifest()
    assert "examples" in m and "frames" in m
