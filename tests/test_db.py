"""Round-trip tests for stats.db against an ISOLATED temp database.

Every test uses the `temp_db` fixture (see conftest.py), which monkeypatches
stats.db.DB_PATH to a tmp_path file and re-runs init_db(). The real
data/hooptracker.db is never opened by these operations.

We verify the full lifecycle:
  create_session -> add_shot(form=...) -> finalize_session -> get_session,
plus list_sessions, all_shots, and delete_session.
"""
import pytest


def test_real_db_path_is_not_used(temp_db):
    """The fixture must have redirected us off the real database file."""
    assert temp_db.DB_PATH.name == "hooptracker_test.db"
    assert "data" not in temp_db.DB_PATH.parts or temp_db.DB_PATH.parent.name != "data"


def test_create_and_get_empty_session(temp_db):
    sid = temp_db.create_session("solo", source="webcam")
    assert isinstance(sid, int)
    obj = temp_db.get_session(sid)
    assert obj is not None
    assert obj["session"]["mode"] == "solo"
    assert obj["session"]["source"] == "webcam"
    assert obj["shots"] == []


def test_shot_form_survives_round_trip(temp_db):
    sid = temp_db.create_session("solo")
    form = {
        "elbow_angle": 172.5,
        "knee_angle": 131.0,
        "lean_deg": 3.2,
        "follow_through": True,
        # an "extra" metric that has NO dedicated column -> must survive via form_json
        "release_angle": 53.7,
        "symmetry_deg": 4.1,
    }
    temp_db.add_shot(sid, "make", t=1.5, zone="top_key", x=12.0, y=34.0, form=form)

    obj = temp_db.get_session(sid)
    assert len(obj["shots"]) == 1
    shot = obj["shots"][0]

    assert shot["result"] == "make"
    assert shot["made"] == 1
    assert shot["zone"] == "top_key"
    # the parsed 'form' dict round-trips fully, including the extra keys
    assert isinstance(shot["form"], dict)
    assert shot["form"]["elbow_angle"] == pytest.approx(172.5)
    assert shot["form"]["release_angle"] == pytest.approx(53.7)
    assert shot["form"]["symmetry_deg"] == pytest.approx(4.1)
    assert shot["form"]["follow_through"] is True


def test_finalize_computes_makes_attempts_fg_pct(temp_db):
    sid = temp_db.create_session("solo")
    temp_db.add_shot(sid, "make", form={"elbow_angle": 170})
    temp_db.add_shot(sid, "miss", form={"elbow_angle": 150})
    temp_db.add_shot(sid, "make", form={"elbow_angle": 168})

    result = temp_db.finalize_session(sid)
    assert result["makes"] == 2
    assert result["attempts"] == 3
    assert result["fg_pct"] == pytest.approx(66.7, abs=0.05)

    # and the persisted session row reflects the same numbers
    obj = temp_db.get_session(sid)
    sess = obj["session"]
    assert sess["makes"] == 2
    assert sess["attempts"] == 3
    assert sess["fg_pct"] == pytest.approx(66.7, abs=0.05)
    assert sess["ended_at"]  # finalize stamps an end time


def test_finalize_empty_session_is_zero_not_error(temp_db):
    sid = temp_db.create_session("solo")
    result = temp_db.finalize_session(sid)
    assert result == {"session_id": sid, "makes": 0, "attempts": 0, "fg_pct": 0.0}


def test_shots_returned_in_insertion_order(temp_db):
    sid = temp_db.create_session("solo")
    for i in range(5):
        temp_db.add_shot(sid, "make" if i % 2 == 0 else "miss",
                         t=float(i), form={"elbow_angle": 160 + i})
    shots = temp_db.get_session(sid)["shots"]
    assert [s["t"] for s in shots] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert [s["form"]["elbow_angle"] for s in shots] == [160, 161, 162, 163, 164]


def test_add_shot_with_no_form_yields_empty_form_dict(temp_db):
    sid = temp_db.create_session("solo")
    temp_db.add_shot(sid, "miss")  # no form kwarg
    shot = temp_db.get_session(sid)["shots"][0]
    # form_json stored as "{}" -> parsed back to {}
    assert shot["form"] == {}
    assert shot["made"] == 0


def test_list_sessions_returns_newest_first(temp_db):
    s1 = temp_db.create_session("first")
    s2 = temp_db.create_session("second")
    sessions = temp_db.list_sessions()
    ids = [s["id"] for s in sessions]
    assert s1 in ids and s2 in ids
    # ORDER BY id DESC -> newest (s2) before s1
    assert ids.index(s2) < ids.index(s1)


def test_list_sessions_respects_limit(temp_db):
    for _ in range(4):
        temp_db.create_session("solo")
    assert len(temp_db.list_sessions(limit=2)) == 2


def test_delete_session_removes_session_and_its_shots(temp_db):
    sid = temp_db.create_session("solo")
    temp_db.add_shot(sid, "make", form={"elbow_angle": 170})
    temp_db.add_shot(sid, "miss", form={"elbow_angle": 150})
    assert temp_db.get_session(sid) is not None

    temp_db.delete_session(sid)

    assert temp_db.get_session(sid) is None
    assert all(s["id"] != sid for s in temp_db.list_sessions())
    # the shots are gone too (no orphans in all_shots)
    assert all(sh["session_id"] != sid for sh in temp_db.all_shots())


def test_get_missing_session_returns_none(temp_db):
    assert temp_db.get_session(999999) is None


def test_all_shots_spans_sessions_and_attaches_form(temp_db):
    s1 = temp_db.create_session("a")
    s2 = temp_db.create_session("b")
    temp_db.add_shot(s1, "make", form={"elbow_angle": 170})
    temp_db.add_shot(s2, "miss", form={"elbow_angle": 150})
    rows = temp_db.all_shots()
    # only our two shots exist in the temp DB
    assert len(rows) == 2
    assert all(isinstance(r["form"], dict) for r in rows)
    sids = {r["session_id"] for r in rows}
    assert sids == {s1, s2}


def test_full_lifecycle_then_consistency_can_consume_it(temp_db):
    """End-to-end: persisted shots feed straight into the consistency engine,
    proving the two modules agree on the shot schema ('made' + 'form')."""
    from stats import consistency as C

    sid = temp_db.create_session("solo")
    elbow = [170, 171, 169, 170, 171, 169]
    made = [True, False, True, False, True, False]
    for e, m in zip(elbow, made):
        temp_db.add_shot(sid, "make" if m else "miss",
                         form={"elbow_angle": e, "lean_deg": (5 if m else 25)})
    temp_db.finalize_session(sid)

    shots = temp_db.get_session(sid)["shots"]
    # db stores made as 0/1; consistency._made handles truthiness + result str
    summary = C.session_consistency(shots)
    assert summary["metrics"]  # metrics computed from the round-tripped forms
    assert "elbow_angle" in summary["metrics"]
