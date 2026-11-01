import gc
import time
import torch
from src.utils import settings as S
from src.utils.official import set_seed
from src.trainer.evaluator import evaluate
from src.trainer.trainer import train
from src.utils.results import export_results, bundle
from src.utils.io import save, initialize


def run(config):
    started = initialize(config)
    cfg = config.newt
    status, completed = 'failed', 0
    try:
        set_seed(0)
        evaluate(cfg, None, S.OUTPUT/'baseline', S.config.final_eval_seeds)
        export_results()
        bundle()
        for seed in S.config.train_seeds:
            branches = ['mpc','prior'] if seed%2==0 else ['prior','mpc']
            for branch in branches:
                folder = S.OUTPUT/f'seed-{seed}-{branch}'
                folder.mkdir()
                torch.cuda.reset_peak_memory_stats()
                train(cfg, folder, branch, seed)
                completed += 1
                gc.collect()
                torch.cuda.empty_cache()
                for index in S.config.checkpoint_rounds:
                    step = index*500*len(S.config.train_tasks)
                    point = folder/f'step-{step:06}'
                    seeds = S.config.final_eval_seeds if index==S.config.learning_rounds else S.config.eval_seeds
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
