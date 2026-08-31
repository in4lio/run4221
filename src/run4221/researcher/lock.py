from __future__ import annotations

import fcntl
import os
from pathlib import Path


class ProcessLock:
    """Advisory single-process lock kept on the persistent data volume."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._descriptor: int | None = None

    def acquire(self) -> bool:
        if self._descriptor is not None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.path, flags, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(descriptor)
            return False
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"{os.getpid()}\n".encode())
        os.fsync(descriptor)
        self._descriptor = descriptor
        return True

    def release(self) -> None:
        if self._descriptor is None:
            return
        descriptor, self._descriptor = self._descriptor, None
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def __enter__(self) -> ProcessLock:
        if not self.acquire():
            raise RuntimeError("Another researcher worker owns the process lock.")
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()
