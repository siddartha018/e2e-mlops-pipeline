from __future__ import annotations

from pathlib import Path

from mlops_pipeline.core import ArtefactStore, FakeClock, content_hash


def test_artefact_store_roundtrip(tmp_path: Path) -> None:
    store = ArtefactStore(root=tmp_path)
    data = b"hello world"
    digest = store.put(data)
    assert store.exists(digest)
    assert store.get(digest) == data
    # Idempotent write: same data -> same hash, no error.
    assert store.put(data) == digest


def test_artefact_store_layout(tmp_path: Path) -> None:
    store = ArtefactStore(root=tmp_path)
    digest = store.put(b"x")
    p = store.path(digest)
    assert p.parent.name == digest[:2]
    assert p.name == digest[2:]


def test_artefact_store_file(tmp_path: Path) -> None:
    src = tmp_path / "src.bin"
    src.write_bytes(b"file contents")
    store = ArtefactStore(root=tmp_path / "store")
    digest = store.put_file(src)
    assert store.get(digest) == b"file contents"


def test_content_hash_stable() -> None:
    a = content_hash("foo", "bar")
    b = content_hash("foo", "bar")
    c = content_hash("foobar")
    assert a == b
    assert a != c  # length-prefix protects against boundary ambiguity


def test_fake_clock() -> None:
    c = FakeClock(start=100.0)
    assert c.now() == 100.0
    c.sleep(5.0)
    assert c.now() == 105.0
    c.advance(10.0)
    assert c.now() == 115.0
