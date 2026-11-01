import gc
import sys
import time
import hashlib
from pathlib import Path
from .official import Config, set_seed
import hydra
import torch
from hydra.core.config_store import ConfigStore
from . import settings as S
from .evaluation import evaluate
from .training import train
from .results import export_results, bundle
from .utils import save

ConfigStore.instance().store(name='config', node=Config)


@hydra.main(version_base=None, config_name='config')
def main(cfg):
    started = time.monotonic()
    S.DEADLINE = started + 11*3600 - 15*60
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    S.OUTPUT.mkdir(parents=True)
    save(S.ROOT/'latest-run.json', dict(folder=str(S.OUTPUT)))
    save(S.OUTPUT/'hardware.json', dict(python=sys.version, torch=torch.__version__,
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
        gpu_bytes=torch.cuda.get_device_properties(0).total_memory,
        cpu_threads=1))
    save(S.OUTPUT/'protocol.json', dict(train_tasks=S.TRAIN_TASKS, ood_tasks=S.OOD_TASKS,
        training_seeds=S.TRAIN_SEEDS, eval_seeds=S.EVAL_SEEDS, final_eval_seeds=S.FINAL_EVAL_SEEDS,
        warmup_rounds=S.WARMUP_ROUNDS, learning_rounds=S.LEARNING_ROUNDS,
        checkpoint_rounds=S.CHECKPOINT_ROUNDS, batch_size=S.BATCH_SIZE, utd=S.UTD,
        replay_capacity=12000, horizon=3, iterations=6, samples=512, elites=64,
        pi_trajectories=24, planner_step=S.PLAN_STEP, budget_seconds=39600,
        model='soup-2M-default.pt',
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob('*.py')}))
    status, completed = 'failed', 0
    try:
        set_seed(0)
        evaluate(cfg, None, S.OUTPUT/'baseline', S.FINAL_EVAL_SEEDS)
        export_results()
        bundle()
        for seed in S.TRAIN_SEEDS:
            branches = ['mpc','prior'] if seed%2==0 else ['prior','mpc']
            for branch in branches:
                folder = S.OUTPUT/f'seed-{seed}-{branch}'
                folder.mkdir()
                torch.cuda.reset_peak_memory_stats()
                train(cfg, folder, branch, seed)
                completed += 1
                gc.collect()
                torch.cuda.empty_cache()
                for index in S.CHECKPOINT_ROUNDS:
                    step = index*500*len(S.TRAIN_TASKS)
                    point = folder/f'step-{step:06}'
                    seeds = S.FINAL_EVAL_SEEDS if index==S.LEARNING_ROUNDS else S.EVAL_SEEDS
                    evaluate(cfg, point/'checkpoint.pt', point, seeds)
                    export_results()
                    bundle()
        status = 'completed'
    except TimeoutError:
        status = 'budget_exhausted'
    except Exception as error:
        save(S.OUTPUT/'error.json', dict(error=str(error)))
        raise
    finally:
        export_results()
        save(S.OUTPUT/'status.json', dict(status=status, completed_training_runs=completed,
             elapsed_seconds=time.monotonic()-started))
        bundle()
        print(f'{status}: {S.ARCHIVE}', flush=True)
