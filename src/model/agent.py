import json
from copy import deepcopy
import torch
from src.utils import settings as S
from src.utils.official import parse_cfg, WorldModel, TDMPC2, discount_heuristic, set_seed, make_env


def make_agent(raw_cfg, tasks, weights=None, seed=0):
    cfg = deepcopy(raw_cfg)
    cfg.task = 'soup'
    cfg.model_size = 'S'
    cfg.obs = 'state'
    cfg.compile = False
    cfg.use_demos = False
    cfg.finetune = False
    cfg.enable_wandb = False
    cfg.save_video = False
    cfg.env_mode = 'sync'
    cfg.seed = seed
    cfg.batch_size = S.config.batch_size
    cfg.lr_schedule = None
    cfg = parse_cfg(cfg)
    cfg.num_envs = len(tasks)
    cfg.obs_shape = {'state': (128,)}
    cfg.action_dim = 16
    cfg.episode_length = 500
    model = WorldModel(cfg).to('cuda:0')
    agent = TDMPC2(model, cfg)
    agent.load(str(S.ROOT / 'checkpoints/soup-2M-default.pt'))
    metadata = json.loads((S.ROOT / 'third_party/newt/tasks.json').read_text())
    added = [name for name in S.config.ood_tasks if name not in cfg.global_tasks]
    embeddings = torch.tensor([metadata[name]['text_embedding'] for name in added], device='cuda:0')
    model._task_emb._parameters['weight'] = torch.cat([model._task_emb.weight, embeddings])
    masks = torch.zeros(len(added), 16, device='cuda:0')
    for index, name in enumerate(added):
        masks[index, :metadata[name]['action_dim']] = 1
        cfg.global_tasks.append(name)
        cfg.task_embeddings.append(metadata[name]['text_embedding'])
        cfg.action_dims.append(metadata[name]['action_dim'])
        cfg.episode_lengths.append(metadata[name]['max_episode_steps'])
        cfg.discounts.append(metadata[name].get('discount_factor', discount_heuristic(cfg, 500)))
    model._action_masks = torch.cat([model._action_masks, masks])
    cfg.num_global_tasks = len(cfg.global_tasks)
    agent.discount = torch.tensor(cfg.discounts, device='cuda:0')
    if weights is not None:
        agent.load(str(weights))
    agent.eval()
    return agent, cfg


def make_task_env(cfg, tasks, seed):
    env_cfg = deepcopy(cfg)
    indices = [cfg.global_tasks.index(name) for name in tasks]
    env_cfg.task = tasks[0]
    env_cfg.tasks = list(tasks)
    env_cfg.global_tasks = list(tasks)
    env_cfg.num_envs = env_cfg.num_tasks = env_cfg.num_global_tasks = len(tasks)
    env_cfg.episode_lengths = [cfg.episode_lengths[i] for i in indices]
    env_cfg.action_dims = [cfg.action_dims[i] for i in indices]
    env_cfg.discounts = [cfg.discounts[i] for i in indices]
    env_cfg.seed = seed
    set_seed(seed)
    env = make_env(env_cfg)
    return env, torch.tensor(indices, dtype=torch.int32), env_cfg
