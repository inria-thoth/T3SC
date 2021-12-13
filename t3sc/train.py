import logging
import os

from omegaconf import errors
import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
import torch

from t3sc import models
from t3sc.data import DataModule
from t3sc.utils import Tester
from t3sc.callbacks import Backtracking

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device in use : {device}")

    # Fix seed for reproducibility
    logger.info(f"Using random seed {cfg.seed}")
    pl.seed_everything(cfg.seed)

    # Load datamodule
    datamodule = DataModule(**cfg.data.params)

    # Logger
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir="tb", name="", version=""
    )

    # Callbacks
    callbacks = [
        cb.ModelCheckpoint(**cfg.checkpoint),
        cb.ModelCheckpoint(**cfg.checkpoint_best),
        cb.LearningRateMonitor(),
        cb.ProgressBar(),
    ]
    try:
        logger.info("Loading backtracking config")
        callbacks.append(Backtracking(**cfg.model.backtracking))
        logger.info("Backtracking callback instantiated successfully")
    except (errors.ConfigAttributeError, TypeError):
        logger.info("Backtracking config not found")

    # Instantiate model
    model_class = models.__dict__[cfg.model.class_name]
    model = model_class(**cfg.model.params).to(device)

    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir="tb", name="", version=""
    )

    # Instantiate trainer
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=tb_logger,
        progress_bar_refresh_rate=0,
        **cfg.trainer.params,
    )

    # Print model info
    model.count_params()

    # Fit trainer
    trainer.fit(model, datamodule=datamodule)

    # Load best checkpoint
    filename_best = os.listdir("best")[0]
    path_best = os.path.join("best", filename_best)
    logger.info(f"Loading best model for testing : {path_best}")
    model.load_state_dict(torch.load(path_best)["state_dict"])

    tester = Tester(**cfg.test)
    tester.eval(model, datamodule=datamodule)
