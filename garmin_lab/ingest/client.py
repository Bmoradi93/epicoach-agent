"""Authenticated, paced, retrying Garmin Connect client.

The vendored ``garminconnect`` client has no backoff or pacing; this module
adds both so that a multi-thousand-request export can run unattended.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from garminconnect import Garmin

log = logging.getLogger(__name__)
T = TypeVar("T")

REPO_ROOT = Path(__file__).resolve().parents[2]
_NON_RETRYABLE = ("404", "400", "403", "Not Found", "Bad Request")


def _is_retryable(exc: BaseException) -> bool:
    msg = str(exc)
    return not any(tok in msg for tok in _NON_RETRYABLE)


def make_client() -> Garmin:
    """Login using the token store if possible, else credentials from .env."""
    load_dotenv(REPO_ROOT / ".env")
    tokdir = os.getenv("GARMINTOKENS", os.path.expanduser("~/.garminconnect"))
    email = os.getenv("GARMIN_EMAIL")
    password = os.getenv("GARMIN_PASSWORD")
    api = Garmin(email, password)
    try:
        api.login(tokdir)
        log.info("Garmin login via token store (%s)", tokdir)
    except Exception as exc:  # noqa: BLE001 - fall back to credential login
        log.warning("Token login failed (%s); logging in with credentials", exc)
        if not (email and password):
            raise RuntimeError("GARMIN_EMAIL / GARMIN_PASSWORD not set in .env")
        api = Garmin(email, password)
        api.login()
        api.garth.dump(tokdir)
        log.info("Fresh login OK; tokens saved to %s", tokdir)
    return api


class Paced:
    """Wrap a client so every call is rate-limited and retried with backoff."""

    def __init__(self, api: Garmin, min_interval_s: float = 0.6) -> None:
        self.api = api
        self.min_interval_s = min_interval_s
        self._last = 0.0
        self.n_calls = 0

    def _pace(self) -> None:
        wait = self.min_interval_s - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=90),
        retry=retry_if_exception(_is_retryable),
    )
    def call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        self._pace()
        self.n_calls += 1
        return fn(*args, **kwargs)
