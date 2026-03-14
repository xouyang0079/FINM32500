# -*- coding: utf-8 -*-
import requests

ALPACA_API_KEY = "PKAQHQZWA2AHHIJPIXYBUG7GZB"
ALPACA_SECRET_KEY = "GPduQrJ8q27Md64bYfonqYeZkuLgc8i8swZqvQYSh5bS"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"


HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
}


def main():
    # 1) Cancel all open orders
    r1 = requests.delete(f"{ALPACA_BASE_URL}/v2/orders", headers=HEADERS, timeout=30)
    print("Cancel orders status:", r1.status_code)
    try:
        print(r1.json())
    except Exception:
        print(r1.text)

    # 2) Liquidate all positions
    r2 = requests.delete(f"{ALPACA_BASE_URL}/v2/positions", headers=HEADERS, timeout=30)
    print("Liquidate positions status:", r2.status_code)
    try:
        print(r2.json())
    except Exception:
        print(r2.text)


if __name__ == "__main__":
    main()