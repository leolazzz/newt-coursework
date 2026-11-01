import json
import hashlib
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parents[1]
revision = '7eef11eb63c8ed53d61d739693d7140135ea0876'
filename = 'soup-2M-default.pt'
path = root / 'checkpoints' / filename
path.parent.mkdir(exist_ok=True)
if not path.exists():
    urllib.request.urlretrieve(
        f'https://huggingface.co/nicklashansen/newt/resolve/{revision}/{filename}', path)
(root / 'checkpoint-manifest.json').write_text(json.dumps(dict(
    revision=revision, filename=filename,
    sha256=hashlib.sha256(path.read_bytes()).hexdigest()), indent=2))
