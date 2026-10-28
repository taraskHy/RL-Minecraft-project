"""Exclusive cache ownership and durable publication for resumable evaluations."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile


@contextmanager
def cache_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as fh:
        fh.seek(0, os.SEEK_END)
        if not fh.tell():
            fh.write(b'0')
            fh.flush()
        fh.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise SystemExit(f'Another process owns {path}; wait for it before retrying') from exc
        try:
            yield
        finally:
            fh.seek(0)
            if os.name == 'nt':
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unique temporary names also allow independent kinds to finish together.
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix=path.name + '.', suffix='.tmp', delete=False) as fh:
        json.dump(value, fh, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
        tmp = Path(fh.name)
    tmp.replace(path)


def append_jsonl_once(path, row):
    """A restart after the append but before its marker must not append twice."""
    path = Path(path)
    with cache_lock(path.with_suffix(path.suffix + '.lock')):
        if path.exists():
            for line in path.read_text().splitlines():
                previous = json.loads(line)  # fail closed on damaged evidence
                if previous.get('run_id') == row['run_id']:
                    if previous != row:
                        raise SystemExit(f'Conflicting published result for {row["run_id"]} in {path}')
                    return
        with path.open('a') as fh:
            fh.write(json.dumps(row) + '\n')
            fh.flush()
            os.fsync(fh.fileno())
