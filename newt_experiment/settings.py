import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path('/kaggle/temp/newt-isolated/runtime')
OUTPUT = ROOT / 'runs' / time.strftime('%Y%m%d-%H%M%S')
ARCHIVE = Path('/kaggle/working/newt-multitask-full-results.zip')
DEADLINE = float('inf')

TRAIN_TASKS = ['walker-stand', 'walker-walk', 'walker-run',
               'hopper-stand', 'hopper-hop', 'cheetah-run']
OOD_TASKS = ['walker-walk-incline', 'spinner-spin-four']
TRAIN_SEEDS = [0, 1, 2]
EVAL_SEEDS = list(range(1000, 1005))
FINAL_EVAL_SEEDS = list(range(1000, 1010))
BATCH_SIZE = 128
WARMUP_ROUNDS = 1
LEARNING_ROUNDS = 100
CHECKPOINT_ROUNDS = [20, 40, 60, 80, 100]
UTD = 0.075
PLAN_STEP = 100_000_000
