import json
import time
from . import settings as S


def save(path, data):
    path.write_text(json.dumps(data, indent=2, default=str, allow_nan=False))


def check_time():
    if time.monotonic() >= S.DEADLINE:
        raise TimeoutError('Time budget reached.')
