import json
import time
import resource
from dataclasses import asdict
import numpy as np
import torch
from tensordict import TensorDict
from src.utils import settings as S
from src.utils.official import Buffer, set_seed
from src.model.agent import make_agent, make_task_env
from src.trainer.rollout import rollout
from src.utils.io import save, check_time
from src.utils.results import bundle


def to_replay(obs, act, rew, task_ids):
    return TensorDict(dict(obs=obs,
        action=torch.cat([torch.zeros_like(act[:1]), act]),
        reward=torch.cat([torch.zeros_like(rew[:1]), rew]),
        task=task_ids.repeat(len(obs), 1)), batch_size=obs.shape[:2])


def train(raw_cfg, folder, branch, seed):
    started = time.perf_counter()
    set_seed(seed)
    agent, cfg = make_agent(raw_cfg, S.config.train_tasks, seed=seed)
    env, ids, env_cfg = make_task_env(cfg, S.config.train_tasks, seed)
    save(folder / 'model-config.json', asdict(cfg))
    save(folder / 'env-config.json', asdict(env_cfg))
    buffer = Buffer(capacity=12000, batch_size=S.config.batch_size, horizon=3, compile=False)
    updates, training_seconds, timings = 0, 0.0, []
    round_steps = 500 * len(S.config.train_tasks)
    try:
        for index in range(S.config.warmup_rounds + S.config.learning_rounds):
            check_time()
            t = time.perf_counter()
            obs, act, rew, timing = rollout(agent, env, ids, branch,
                folder / f'rollout-{index:03}.npz', training=True,
                random_actions=index < S.config.warmup_rounds)
            data = to_replay(obs, act, rew, ids)
            for i in range(len(S.config.train_tasks)):
                buffer.add(data[:, i].unsqueeze(0))
            count = 0 if index < S.config.warmup_rounds else int(round_steps * S.config.utd)
            torch.cuda.synchronize()
            update_start = time.perf_counter()
            with (folder / 'updates.jsonl').open('a') as log:
                for _ in range(count):
                    metrics = {k:float(v.item()) for k,v in agent.update(buffer).items()}
                    if not all(np.isfinite(v) for v in metrics.values()):
                        raise RuntimeError('Non-finite training loss.')
                    updates += 1
                    log.write(json.dumps(dict(update=updates, **metrics))+'\n')
            torch.cuda.synchronize()
            timing.update(round=index, updates=count,
                          update_seconds=time.perf_counter()-update_start)
            timings.append(timing)
            training_seconds += time.perf_counter()-t
            save(folder / 'timing.json', timings)
            agent.save(str(folder / 'checkpoint.pt'))
            step = index * round_steps
            save(folder / 'training.json', dict(training_seed=seed, branch=branch,
                online_transitions=step, updates=updates, training_seconds=training_seconds,
                elapsed_seconds=time.perf_counter()-started,
                peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
            print(f'{folder.name}: {step}/{S.config.learning_rounds*round_steps}, updates={updates}', flush=True)
            if index in S.config.checkpoint_rounds:
                point = folder / f'step-{step:06}'
                point.mkdir()
                torch.save({'model':agent.model.state_dict()}, point/'checkpoint.pt')
                save(point/'info.json', dict(branch=branch, training_seed=seed,
                    online_transitions=step, training_seconds=training_seconds))
                bundle()
    finally:
        env.close()
