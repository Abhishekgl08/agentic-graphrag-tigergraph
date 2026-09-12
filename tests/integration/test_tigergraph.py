import os
from pathlib import Path

import pyTigerGraph as tg
from dotenv import load_dotenv
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)


def test_tigergraph_connection():
    host = os.getenv("TG_HOST")
    secret = os.getenv("TG_SECRET")
    graphname = os.getenv("TG_GRAPHNAME")

    assert host, f"TG_HOST is missing. Expected .env at: {ENV_FILE}"
    assert secret, f"TG_SECRET is missing. Expected .env at: {ENV_FILE}"
    assert graphname, f"TG_GRAPHNAME is missing. Expected .env at: {ENV_FILE}"

    print("\n--- Configuration ---")
    print(f"Host: {host}")
    print(f"Graph: {graphname}")
    print(f"Secret loaded: {bool(secret)}")

    conn = tg.TigerGraphConnection(
        host=host,
        graphname=graphname,
        gsqlSecret=secret,
    )

    try:
        token = conn.getToken(secret)

        assert token, "TigerGraph returned an empty token"

        print("\nTigerGraph authentication successful.")
        print("Token received: YES")

    except requests.exceptions.HTTPError as exc:
        print("\n--- TigerGraph authentication failed ---")

        response = exc.response

        if response is not None:
            print(f"HTTP status: {response.status_code}")
            print(f"Response URL: {response.url}")
            print("Response body:")
            print(response.text)

        raise