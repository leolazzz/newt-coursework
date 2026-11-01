from dataclasses import asdict
from src.utils import settings as S
from src.model.agent import make_agent, make_task_env
from src.trainer.rollout import rollout
from src.utils.io import save, check_time


def evaluate(raw_cfg, weights, folder, seeds):
    folder.mkdir(parents=True, exist_ok=True)
    tasks = S.config.train_tasks + S.config.ood_tasks
    agent, cfg = make_agent(raw_cfg, tasks, weights)
    rows = []
    for seed in seeds:
        for mode in ['prior', 'mpc']:
            check_time()
            env, ids, env_cfg = make_task_env(cfg, tasks, seed)
            try:
                obs, act, rew, timing = rollout(agent, env, ids, mode,
                    folder / f'eval-{seed}-{mode}.npz')
                for i, task in enumerate(tasks):
                    rows.append(dict(task=task, split='ood' if task in S.config.ood_tasks else 'train',
                        mode=mode, seed=seed, steps=len(rew), **{'return':timing['returns'][i]},
                        score=timing['returns'][i]/1000,
                        batch_episode_seconds=timing['collection_seconds'],
                        batch_action_seconds_mean=timing['action_seconds']/len(rew)))
                save(folder / 'evaluation.json', rows)
                print(f'{folder.name}: evaluated seed={seed} {mode}', flush=True)
            finally:
                env.close()
    save(folder / 'evaluation.json', rows)
    save(folder / 'config.json', asdict(cfg))
