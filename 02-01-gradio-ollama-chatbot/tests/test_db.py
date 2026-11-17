from supportbot import db


def test_log_and_read_back_newest_first():
    db.log_turn("s1", "llama3.2", "q1", "a1")
    db.log_turn("s1", "llama3.2", "q2", "a2")
    rows = db.recent_turns()
    assert [r[3] for r in rows] == ["q2", "q1"]
    assert rows[0][1:] == ["s1", "llama3.2", "q2", "a2"]


def test_limit_is_respected():
    for i in range(5):
        db.log_turn("s", "m", f"q{i}", "a")
    assert len(db.recent_turns(limit=3)) == 3


def test_sql_injection_is_stored_as_plain_text():
    evil = "'); DROP TABLE chat_log; --"
    db.log_turn("s", "m", evil, "a")
    assert db.recent_turns()[0][3] == evil
