# Goal: Convert raw FIX protocol strings into structured Python dictionaries.

from __future__ import annotations


class FixParser:
    def parse(self, msg: str):
        if not isinstance(msg, str) or not msg.strip():
            raise ValueError("FIX message must be a non-empty string")

        fields: dict[str, str] = {}
        parts = msg.strip().split("|")

        for part in parts:
            if not part:
                continue
            if "=" not in part:
                raise ValueError(f"Bad FIX field (missing '='): {part}")
            tag, value = part.split("=", 1)
            tag = tag.strip()
            value = value.strip()
            if not tag:
                raise ValueError(f"Bad FIX field (empty tag): {part}")
            fields[tag] = value

        self._validate(fields)
        return fields

    def _validate(self, f: dict[str, str]):
        # validate required fields by message type
        msg_type = f.get("35")
        if not msg_type:
            raise ValueError("Missing required tag: 35 (MsgType)")

        if msg_type == "D":  # New Order - Single
            required = ["55", "54", "38", "40"]  # symbol, side, qty, ordType
            for tag in required:
                if tag not in f or f[tag] == "":
                    raise ValueError(f"Missing required tag: {tag}")

            # validate price for limit order
            if f.get("40") == "2" and (f.get("44", "").strip() == ""):
                raise ValueError("Missing required tag: 44 (Price) for limit order")

        elif msg_type == "S":
            required = ["55"]
            for tag in required:
                if tag not in f or f[tag] == "":
                    raise ValueError(f"Missing required tag: {tag}")

        else:
            return


if __name__ == "__main__":
    msg = "8=FIX.4.2|35=D|55=AAPL|54=1|38=100|40=2|44=189.5|10=128"
    print(FixParser().parse(msg))