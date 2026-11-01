import json
import hashlib
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parents[1]
revision = '7eef11eb63c8ed53d61d739693d7140135ea0876'
filename = 'soup-2M-default.pt'
path = root / 'checkpoints' / filename
manifest = root / 'checkpoint-manifest.json'
path.parent.mkdir(exist_ok=True)
if not (path.exists() and manifest.exists()):
    url = f'https://huggingface.co/nicklashansen/newt/resolve/{revision}/{filename}'
    partial = path.with_suffix('.part')
    print(f'Downloading {filename}...', flush=True)
    with urllib.request.urlopen(url, timeout=60) as response, partial.open('wb') as output:
        total = int(response.headers.get('Content-Length', 0))
        downloaded = 0
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            downloaded += len(chunk)
            print(f'Checkpoint: {downloaded / 1024**2:.1f} MiB downloaded', flush=True)
    if total and downloaded != total:
        raise RuntimeError(f'Incomplete checkpoint: {downloaded}/{total} bytes')
    partial.replace(path)
manifest.write_text(json.dumps(dict(
    revision=revision, filename=filename,
    sha256=hashlib.sha256(path.read_bytes()).hexdigest()), indent=2))
print(f'Checkpoint ready: {path.stat().st_size / 1024**2:.1f} MiB', flush=True)
