"""Durable path checkpoints and cooperative pause requests."""
from __future__ import annotations
import hashlib
import fcntl
import time
from contextlib import contextmanager
import json
import os
from pathlib import Path


# Strategy semantics changed: old paths and optimizer scores can contain
# uncapped inventory. Keep those files, but recompute instead of reusing them.
ENGINE_REVISION = 'inventory-cap-v1'


def engine_folder(folder: Path | None) -> Path | None:
    return folder / ENGINE_REVISION if folder else None


class SimulationPaused(Exception):
    pass


def root() -> Path | None:
    value = os.environ.get('TQQQ_CHECKPOINT_DIR')
    return Path(value) if value else None


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def check_pause() -> None:
    folder = root()
    if folder and (folder / 'pause.requested').exists():
        raise SimulationPaused('Paused after the last completed path')


class PathCheckpoint:
    def __init__(self, folder: Path | None, config: dict):
        self.folder = engine_folder(folder)
        if folder:
            fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
            manifest = folder / 'manifest.json'
            if manifest.exists() and json.loads(manifest.read_text())['fingerprint'] != fingerprint:
                raise ValueError('Checkpoint inputs differ from the requested configuration')
            atomic_json(manifest, {'fingerprint': fingerprint})

    def load(self, index: int):
        check_pause()
        path = self.folder / f'path-{index}.json' if self.folder else None
        return json.loads(path.read_text()) if path and path.exists() else None

    def save(self, index: int, value: dict):
        if self.folder:
            atomic_json(self.folder / f'path-{index}.json', value)


@contextmanager
def execution_lock():
    folder = root()
    if not folder:
        yield
        return
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.lock').open('w') as handle:
        while True:
            check_pause()
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(0.1)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def guarded(main):
    try:
        with execution_lock():
            main()
    except SimulationPaused:
        print('TQQQ_PROGRESS ' + json.dumps({'phase': 'Paused at a completed path'}), flush=True)
        raise SystemExit(75)
