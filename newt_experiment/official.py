import os
import sys
from .settings import RUNTIME

os.environ.update(MUJOCO_GL='egl', WANDB_MODE='disabled', WANDB_DISABLED='true',
                  HF_HUB_DISABLE_TELEMETRY='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
                  DO_NOT_TRACK='1')
sys.path.insert(0, str(RUNTIME / 'tdmpc2'))

from config import Config, parse_cfg
from common import set_seed
from common.buffer import Buffer
from common.world_model import WorldModel
from common.math import discount_heuristic
from envs import make_env
from tdmpc2 import TDMPC2
