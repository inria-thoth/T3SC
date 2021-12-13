import logging
import hydra
from omegaconf import DictConfig
import os

from t3sc.train import train
from t3sc.test import test

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@hydra.main(config_path="t3sc/config", config_name="config")
def main(cfg: DictConfig) -> None:

    logger.info(f"Current working directory : {os.getcwd()}")

    try:
        if cfg.mode.name == "train":
            train(cfg)
        else:
            test(cfg)
    except Exception as e:
        logger.critical(e, exc_info=True)


if __name__ == "__main__":
    main()
