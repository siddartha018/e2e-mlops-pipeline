"""Core injectable primitives: Clock, Rng, ArtefactStore.

Every pipeline component takes these via constructor injection so tests can
substitute deterministic fakes.
"""
from __future__ import annotations

import hashlib
import random
import time
from abc import ABC, abstractmethod
from pathlib import Path


class Clock(ABC):
    @abstractmethod
    def now(self) -> float: ...

    @abstractmethod
    def sleep(self, seconds: float) -> None: ...


class RealClock(Clock):
    def now(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class FakeClock(Clock):
    def __init__(self, start: float = 0.0) -> None:
        self._t = start

    def now(self) -> float:
        return self._t

    def sleep(self, seconds: float) -> None:
        self._t += seconds

    def advance(self, seconds: float) -> None:
        self._t += seconds


class Rng(ABC):
    @abstractmethod
    def random(self) -> float: ...

    @abstractmethod
    def randint(self, a: int, b: int) -> int: ...

    @abstractmethod
    def seed(self, s: int) -> None: ...


class RealRng(Rng):
    def __init__(self, seed: int = 0xC0FFEE) -> None:
        self._r = random.Random(seed)

    def random(self) -> float:
        return self._r.random()

    def randint(self, a: int, b: int) -> int:
        return self._r.randint(a, b)

    def seed(self, s: int) -> None:
        self._r = random.Random(s)


class ArtefactStore:
    """Content-addressed blob store on local disk.

    Layout: <root>/blobs/<ab>/<cdef...> where sha256 = 'abcdef...'.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.blobs = self.root / "blobs"
        self.blobs.mkdir(parents=True, exist_ok=True)

    def put(self, data: bytes) -> str:
        digest = hashlib.sha256(data).hexdigest()
        p = self._path(digest)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.rename(p)
        return digest

    def put_file(self, src: Path) -> str:
        h = hashlib.sha256()
        with open(src, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        digest = h.hexdigest()
        dst = self._path(digest)
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(".tmp")
            tmp.write_bytes(Path(src).read_bytes())
            tmp.rename(dst)
        return digest

    def get(self, digest: str) -> bytes:
        return self._path(digest).read_bytes()

    def exists(self, digest: str) -> bool:
        return self._path(digest).exists()

    def path(self, digest: str) -> Path:
        return self._path(digest)

    def _path(self, digest: str) -> Path:
        return self.blobs / digest[:2] / digest[2:]


def content_hash(*parts: bytes | str) -> str:
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, str):
            p = p.encode("utf-8")
        h.update(len(p).to_bytes(8, "big"))
        h.update(p)
    return h.hexdigest()
