import json
from logger import Logger


def test_singleton():
    a = Logger(path="events_a.json")
    b = Logger(path="events_b.json")
    assert a is b  # same instance


def test_save(tmp_path):
    path = tmp_path / "events.json"
    log = Logger(path=str(path))
    log.events = []  # reset for test
    log.log("TestEvent", {"x": 1})
    log.save()

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data[0]["type"] == "TestEvent"
    assert data[0]["data"]["x"] == 1