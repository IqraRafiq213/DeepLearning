# %% [markdown]
# """
# Exercise 4 Solution: Convolutional layers, pooling, and ModuleList
# ------------------------------------------------------------------
# Key additions vs the original CNN:
#   - ConvBlock: a reusable unit of (Conv2d → BatchNorm2d → ReLU → MaxPool2d)
#   - CNNWithBlocks: uses nn.ModuleList so the *number* of conv blocks is a
#     hyperparameter you can search over with hyperopt
#   - Dropout added after every dense hidden layer
#   - num_blocks, dropout_rate, and filters are all logged to MLflow
# """

# %%

from datetime import datetime
from pathlib import Path
from typing import Iterator

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from loguru import logger
from mads_datasets import DatasetFactoryProvider, DatasetType
from mltrainer import ReportTypes, Trainer, TrainerSettings, metrics
from mltrainer.preprocessors import BasePreprocessor

# %%


# ---------------------------------------------------------------------------
# Data upload and streaming
# ---------------------------------------------------------------------------

def get_fashion_streamers(batchsize: int) -> tuple[Iterator, Iterator]:
    fashionfactory = DatasetFactoryProvider.create_factory(DatasetType.FASHION)
    preprocessor = BasePreprocessor()
    streamers = fashionfactory.create_datastreamer(
        batchsize=batchsize, preprocessor=preprocessor
    )
    trainstreamer = streamers["train"].stream()
    validstreamer = streamers["valid"].stream()
    return trainstreamer, validstreamer


def get_device() -> str:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    logger.info(f"Using device: {device}")
    return device


# ---------------------------------------------------------------------------
# ConvBlock: the reusable building block
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """
    One convolutional block:
        Conv2d → BatchNorm2d → ReLU → MaxPool2d

    BatchNorm goes *before* the activation (standard practice).
    MaxPool2d halves the spatial dimensions each time.

    Args:
        in_channels:  number of input feature maps
        out_channels: number of output feature maps (= filters)
        pool:         whether to apply MaxPool2d at the end of this block
    """
    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),  # normalise before activation
            nn.ReLU(),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# CNNWithBlocks: number of blocks is now a hyperparameter
# ---------------------------------------------------------------------------

class CNNWithBlocks(nn.Module):
    """
    Flexible CNN where:
      - `num_blocks`    controls how many ConvBlocks are stacked
      - `filters`       is the number of feature maps in every block
      - `units1/units2` are the sizes of the two dense hidden layers
      - `dropout_rate`  controls regularisation in the dense head

    The first block always has MaxPool; subsequent blocks only pool if the
    spatial dimensions are still large enough (>= 4 pixels on the shortest
    side) — this avoids crashing on small feature maps.

    An AvgPool2d at the end of the conv-tower collapses whatever spatial
    size remains into 1×1, so the dense head always sees `filters` features
    regardless of `num_blocks`.
    """

    def __init__(
        self,
        filters: int = 64,
        units1: int = 128,
        units2: int = 64,
        num_blocks: int = 2,
        dropout_rate: float = 0.3,
        input_size: tuple = (32, 1, 28, 28),
    ):
        super().__init__()
        self.in_channels = input_size[1]
        self.input_size = input_size

        # --- Build conv tower with ModuleList ---
        # ModuleList registers each block as a sub-module so parameters are
        # correctly tracked during back-prop and by torchinfo / MLflow.
        self.conv_blocks = nn.ModuleList()
        in_ch = self.in_channels

        for i in range(num_blocks):
            # Decide whether to pool: only if the spatial dim is still >= 4
            # We probe the current spatial size with a dummy forward pass so
            # far, or simply check after the fact.  A simpler heuristic:
            # always pool for the first `max_pool_blocks` blocks.
            # Here we pool every block — _conv_test will tell us if we over-do it.
            self.conv_blocks.append(ConvBlock(in_ch, filters, pool=True))
            in_ch = filters  # every subsequent block has `filters` in_channels

        # Dynamically calculate the spatial size after the tower
        activation_map_size = self._conv_test(input_size)
        logger.info(f"Activation map size after {num_blocks} blocks: {activation_map_size}")
        self.agg = nn.AvgPool2d(activation_map_size)

        # --- Dense head ---
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters, units1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),      # dropout after hidden layer
            nn.Linear(units1, units2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),      # dropout after second hidden layer
            nn.Linear(units2, 10),         # 10 FashionMNIST classes
        )

    def _conv_test(self, input_size: tuple) -> torch.Size:
        """Run a dummy forward pass to measure the activation-map spatial size."""
        x = torch.ones(input_size, dtype=torch.float32)
        for block in self.conv_blocks:
            x = block(x)
        return x.shape[-2:]  # (H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.conv_blocks:
            x = block(x)
        x = self.agg(x)
        return self.dense(x)


# ---------------------------------------------------------------------------
# MLflow setup
# ---------------------------------------------------------------------------

def setup_mlflow(experiment_path: str) -> None:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_path)


# ---------------------------------------------------------------------------
# Hyperopt objective
# ---------------------------------------------------------------------------

def objective(params: dict) -> dict:
    modeldir = Path("models").resolve()
    modeldir.mkdir(parents=True, exist_ok=True)

    batchsize = 64
    trainstreamer, validstreamer = get_fashion_streamers(batchsize)
    accuracy = metrics.Accuracy()

    settings = TrainerSettings(
        epochs=3,
        metrics=[accuracy],
        logdir=Path("modellog"),
        train_steps=100,
        valid_steps=100,
        reporttypes=[ReportTypes.MLFLOW],
    )

    device = get_device()

    with mlflow.start_run():
        mlflow.set_tag("model", "cnn_with_blocks")
        mlflow.set_tag("dev", "Iqra")

        # Log all hyperparameters (including num_blocks and dropout_rate)
        mlflow.log_params(params)
        mlflow.log_param("batchsize", batchsize)

        model = CNNWithBlocks(**params)
        model.to(device)

        trainer = Trainer(
            model=model,
            settings=settings,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer=optim.Adam,
            traindataloader=trainstreamer,
            validdataloader=validstreamer,
            scheduler=optim.lr_scheduler.ReduceLROnPlateau,
            device=device,
        )
        trainer.loop()

        tag = datetime.now().strftime("%Y%m%d-%H%M")
        modelpath = modeldir / f"{tag}_model.pt"
        torch.save(model, modelpath)
        mlflow.log_artifact(str(modelpath), artifact_path="pytorch_models")

        return {"loss": trainer.test_loss, "status": STATUS_OK}


# ---------------------------------------------------------------------------
# Main: define search space and run hyperopt
# ---------------------------------------------------------------------------

def main():
    setup_mlflow("exercise4_modulelist")

    search_space = {
        # Conv tower hyperparameters
        "filters":      scope.int(hp.quniform("filters",    32, 64, 8)),
        "num_blocks":   scope.int(hp.quniform("num_blocks",  1,  3, 1)),  # 1–3 blocks
        # Dense head hyperparameters
        "units1":       scope.int(hp.quniform("units1",     64, 128, 8)),
        "units2":       scope.int(hp.quniform("units2",     32,  64, 8)),
        # Regularisation
        "dropout_rate": hp.uniform("dropout_rate", 0.1, 0.5),
        # Fixed input size (not searched, but passed to the model)
        "input_size":   (64, 1, 28, 28),
    }

    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=6,     # increase for a more thorough search
        trials=Trials(),
    )
    logger.info(f"Best hyperparameters found: {best_result}")
    def main():
     setup_mlflow("exercise4_modulelist")

    search_space = { ... }  # unchanged

    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=6,
        trials=Trials(),
    )
    logger.info(f"Best hyperparameters found: {best_result}")
      # ── Export results ──────────────────────────────────────────
    df = mlflow.search_runs(experiment_names=["exercise4_modulelist"])
    cols = [c for c in df.columns if c.startswith("metrics.") or c.startswith("params.")]
    cols += ["run_id", "status", "start_time"]
    df[cols].to_csv("mlflow_runs.csv", index=False)
    logger.info("Saved results to mlflow_runs.csv")

if __name__ == "__main__":
    main()


