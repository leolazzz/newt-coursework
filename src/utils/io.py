import json
import time
from src.utils import settings as S


def save(path, data):
    path.write_text(json.dumps(data, indent=2, default=str, allow_nan=False))


def check_time():
    if time.monotonic() >= S.DEADLINE:
        raise TimeoutError('Time budget reached.')


def initialize(config):
    import sys
    import torch
    from omegaconf import OmegaConf

    started = time.monotonic()
    S.config = config.experiment
    S.DEADLINE = started + S.config.budget_hours*3600 - 15*60
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    S.OUTPUT.mkdir(parents=True)
    save(S.ROOT/'latest-run.json', dict(folder=str(S.OUTPUT)))
    OmegaConf.save(config, S.OUTPUT/'config.yaml')
    save(S.OUTPUT/'hardware.json', dict(python=sys.version, torch=torch.__version__,
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
        gpu_bytes=torch.cuda.get_device_properties(0).total_memory, cpu_threads=1))
    return started
