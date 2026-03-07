# Goal: Record system activity for replay and analysis.

from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional


class Logger:
    _instance: Optional["Logger"] = None

    def __new__(cls, path: str = "events.json"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_done = False
        return cls._instance

    def __init__(self, path: str = "events.json"):
        # always update path
        self.path = path

        # initialize event store once
        if getattr(self, "_init_done", False):
            return

        self.events: List[Dict[str, Any]] = []
        self._init_done = True

    def log(self, event_type: str, data: dict):
        event = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "type": str(event_type),
            "data": data,
        }
        self.events.append(event)
        print(f"[LOG] {event_type} → {data}")

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.events, f, indent=2, ensure_ascii=False)