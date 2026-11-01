import time
import hydra
from hydra.core.config_store import ConfigStore
from src.utils.official import Config
from src.utils import settings as S
from src.utils.io import initialize, save
from src.trainer.evaluator import evaluate
from src.utils.results import export_results, bundle

ConfigStore.instance().store(name='newt_base', node=Config)


@hydra.main(version_base=None, config_path='src/configs', config_name='inference')
def main(config):
    started = initialize(config)
    folder = S.OUTPUT / ('baseline' if config.checkpoint is None else 'evaluation')
    folder.mkdir()
    if config.checkpoint is not None:
        save(folder/'info.json', dict(branch='checkpoint', training_seed=None,
             online_transitions=None, training_seconds=None))
    status = 'failed'
    try:
        evaluate(config.newt, config.checkpoint, folder, S.config.final_eval_seeds)
        status = 'completed'
    except TimeoutError:
        status = 'budget_exhausted'
    finally:
        export_results()
        save(S.OUTPUT/'status.json', dict(status=status, elapsed_seconds=time.monotonic()-started))
        bundle()


if __name__ == '__main__':
    main()
