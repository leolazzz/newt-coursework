import hydra
from hydra.core.config_store import ConfigStore
from src.utils.official import Config
from src.trainer.experiment import run

ConfigStore.instance().store(name='newt_base', node=Config)


@hydra.main(version_base=None, config_path='src/configs', config_name='baseline')
def main(config):
    run(config)


if __name__ == '__main__':
    main()
