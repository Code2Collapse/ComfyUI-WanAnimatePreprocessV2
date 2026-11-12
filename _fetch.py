"""Resumable model-file download for this pack (L5.08 slice 2). Stdlib only.

The loaders used to call ``urllib.request.urlretrieve`` straight into a ``.part`` file: a dropped connection threw
the bytes away (the NLF model is hundreds of MB), nothing retried, Cancel in ComfyUI did not stop it, and nothing
checked the size or hash. ``fetch_file`` keeps the ``.part`` between attempts and resumes it with a Range request,
retries transient failures, honours ComfyUI's Cancel between chunks, checks size / SHA-256, and only renames the file
into place when it is complete. Permission is still decided by ``_download_policy`` (the caller checks it first).
"""
from __future__ import annotations

import hashlib
import http.client
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Optional

log = logging.getLogger("WanAnimatePreprocessV2.fetch")
CHUNK = 1 << 20
_UA = "ComfyUI-WanAnimatePreprocessV2"


def _check_interrupt() -> None:
    try:
        import comfy.model_management as mm  # type: ignore
    except Exception:
        return
    mm.throw_exception_if_processing_interrupted()


class _Retry(Exception):
    pass


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def _attempt(url: str, part: str, headers: dict, timeout: float) -> None:
    have = os.path.getsize(part) if os.path.exists(part) else 0
    hdrs = {"User-Agent": _UA, **headers}
    if have:
        hdrs["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=hdrs)
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and have:                     # the .part is already the whole file (or bogus): restart
            os.remove(part)
            raise _Retry("range not satisfiable; restarting") from exc
        if exc.code == 429 or exc.code >= 500:
            raise _Retry(f"HTTP {exc.code}") from exc
        raise RuntimeError(f"HTTP {exc.code} for {url}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise _Retry(str(exc)) from exc
    with resp:
        status = getattr(resp, "status", 200)
        if have and status == 200:                       # server ignored Range: start over
            have = 0
        mode = "ab" if (have and status == 206) else "wb"
        length = resp.headers.get("Content-Length")
        expected = (have if mode == "ab" else 0) + int(length) if length and length.isdigit() else None
        try:
            with open(part, mode) as fh:
                while True:
                    _check_interrupt()
                    b = resp.read(CHUNK)
                    if not b:
                        break
                    fh.write(b)
        except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:   # IncompleteRead = a drop
            raise _Retry(f"connection dropped: {exc}") from exc
    if expected is not None and os.path.getsize(part) != expected:
        raise _Retry(f"got {os.path.getsize(part)} of {expected} bytes")


def fetch_file(url: str, dest: str, *, sha256: Optional[str] = None, min_size: int = 0, retries: int = 3,
               timeout: float = 60, headers: Optional[dict] = None) -> str:
    """Download ``url`` to ``dest`` (via ``dest + '.part'``, resumed when present). Returns ``dest``.

    Raises RuntimeError (HTTP / network failure after ``retries`` retries, file too small) or ValueError (SHA-256
    mismatch; the partial file is deleted). ComfyUI's Cancel raises its own interrupt exception and keeps the
    ``.part`` so the next run resumes."""
    os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)
    part = dest + ".part"
    want = (sha256 or "").strip().lower() or None
    last = ""
    for attempt in range(retries + 1):
        try:
            _attempt(url, part, dict(headers or {}), timeout)
            break
        except _Retry as exc:
            last = str(exc)
            if attempt == retries:
                raise RuntimeError(f"download of {url} failed after {retries + 1} tries: {last}") from None
            wait = 2 ** attempt
            log.warning("download of %s interrupted (%s); retrying in %ss", url, last, wait)
            time.sleep(wait)
    size = os.path.getsize(part)
    if size < min_size:
        os.remove(part)
        raise RuntimeError(f"download of {url} is only {size} bytes (expected at least {min_size}); "
                           "the server probably sent an error page")
    if want:
        got = _sha256(part)
        if got != want:
            os.remove(part)
            raise ValueError(f"SHA-256 mismatch for {url}: expected {want}, got {got}; the file was deleted")
    os.replace(part, dest)
    return dest
