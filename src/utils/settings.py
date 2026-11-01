import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = Path('/kaggle/temp/newt-isolated/runtime')
OUTPUT = ROOT / 'runs' / time.strftime('%Y%m%d-%H%M%S')
ARCHIVE = Path('/kaggle/working/newt-multitask-full-results.zip')
DEADLINE = float('inf')
config = None
