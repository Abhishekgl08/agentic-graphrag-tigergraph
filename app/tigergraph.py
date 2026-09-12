import os
from pathlib import Path

from dotenv import load_dotenv
import pyTigerGraph as tg


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)


def create_connection():
    """Create an authenticated TigerGraph connection."""

    host = os.getenv("TG_HOST")
    graphname = os.getenv("TG_GRAPHNAME")
    secret = os.getenv("TG_SECRET")

    if not host:
        raise RuntimeError("TG_HOST is not set in .env")

    if not graphname:
        raise RuntimeError("TG_GRAPHNAME is not set in .env")

    if not secret:
        raise RuntimeError("TG_SECRET is not set in .env")

    return tg.TigerGraphConnection(
        host=host,
        graphname=graphname,
        gsqlSecret=secret,
    )