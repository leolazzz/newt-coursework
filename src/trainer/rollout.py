import time
import numpy as np
import torch
from src.utils import settings as S


def rollout(agent, env, task_ids, mode, path, training=False, random_actions=False):
    agent.eval()
    agent._prev_mean.zero_()
    obs, _ = env.reset()
    observations, actions, rewards, action_times = [obs], [], [], []
    done = torch.ones(len(task_ids), dtype=torch.bool)
    started = time.perf_counter()
    with torch.no_grad():
        for _ in range(500):
            torch.cuda.synchronize()
            t = time.perf_counter()
            action = env.rand_act() if random_actions else agent(
                obs, t0=done, step=S.config.planner_step, task=task_ids,
                eval_mode=not training, mpc=mode == 'mpc')
            torch.cuda.synchronize()
            action_times.append(time.perf_counter() - t)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated
            observations.append(info['final_observation'] if done.all() else obs)
            actions.append(action)
            rewards.append(reward)
            if done.all():
                break
    if not done.all():
        raise RuntimeError('Episode did not finish.')
    elapsed = time.perf_counter()-started
    obs, act, rew = torch.stack(observations), torch.stack(actions), torch.stack(rewards)
    np.savez_compressed(path, observation=obs.numpy(), action=act.numpy(),
                        reward=rew.numpy(), action_seconds=np.asarray(action_times))
    timing = dict(collection_seconds=elapsed,
                  action_seconds=sum(action_times), returns=rew.sum(0).tolist())
    return obs, act, rew, timing
