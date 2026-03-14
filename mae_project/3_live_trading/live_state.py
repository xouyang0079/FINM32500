# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


STATUS_STARTED = "STARTED"
STATUS_DATA_LOCKED = "DATA_LOCKED"
STATUS_PREDICTED = "PREDICTED"
STATUS_WEIGHTS_COMPUTED = "WEIGHTS_COMPUTED"
STATUS_ORDERS_SUBMITTED = "ORDERS_SUBMITTED"
STATUS_DONE = "DONE"
STATUS_FAILED = "FAILED"


@dataclass
class RunRecord:
    strategy_id: str
    trade_date: str          # ET date when execution is intended
    signal_date: str         # previous trading day used for model input
    status: str
    created_at_utc: str
    updated_at_utc: str
    payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LiveStateStore:
    """
    JSON state store for idempotent daily runs and recovery-aware status tracking.
    """

    def __init__(self, path: str):
        self.path = path
        self._data = self._load()

    def _load(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            return {"runs": {}}
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    # def _save(self) -> None:
    #     os.makedirs(os.path.dirname(self.path), exist_ok=True)
    #     tmp = self.path + ".tmp"
    #     with open(tmp, "w", encoding="utf-8") as f:
    #         json.dump(self._data, f, ensure_ascii=False, indent=2)
    #     os.replace(tmp, self.path)

    def _save(self) -> None:
        import time

        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"

        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())

        last_err = None
        for _ in range(10):
            try:
                os.replace(tmp, self.path)
                return
            except PermissionError as e:
                last_err = e
                time.sleep(0.2)

        raise last_err


    @staticmethod
    def _key(strategy_id: str, trade_date: str) -> str:
        return f"{strategy_id}::{trade_date}"

    def get_run(self, strategy_id: str, trade_date: str) -> Optional[RunRecord]:
        key = self._key(strategy_id, trade_date)
        raw = self._data.get("runs", {}).get(key)
        if raw is None:
            return None
        return RunRecord(**raw)

    def upsert_run(self, rec: RunRecord) -> None:
        key = self._key(rec.strategy_id, rec.trade_date)
        self._data.setdefault("runs", {})[key] = rec.to_dict()
        self._save()

    def patch_run(
        self,
        strategy_id: str,
        trade_date: str,
        *,
        updated_at_utc: str,
        status: Optional[str] = None,
        payload_updates: Optional[Dict[str, Any]] = None,
    ) -> RunRecord:
        rec = self.get_run(strategy_id, trade_date)
        if rec is None:
            raise KeyError(f"Run record not found: {strategy_id} / {trade_date}")

        if status is not None:
            rec.status = status
        rec.updated_at_utc = updated_at_utc

        if payload_updates:
            rec.payload.update(payload_updates)

        self.upsert_run(rec)
        return rec