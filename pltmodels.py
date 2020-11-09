"""
Optuna example that optimizes multivariate Gaussian models using PyTorch Lightning.
"""

import argparse
import os
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pl_bolts.datamodules import SklearnDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

import optuna
from optuna.integration import PyTorchLightningPruningCallback

PERCENT_VALID_EXAMPLES = 0.1
BATCHSIZE = 128
EPOCHS = 10
DIR = os.getcwd()
MODEL_DIR = os.path.join(DIR, "result")


def load_dataset(dataset_path: str = "/home/alizaidi/bonsai/repsol/data/scenario1"):

    X = np.load(os.path.join(dataset_path, "x_set.npy"))
    y = np.load(os.path.join(dataset_path, "y_set.npy"))

    return X, y


X, y = load_dataset()
input_shape = X.shape[1]
output_shape = y.shape[1]
dataset = SklearnDataset(X, y)


mse_loss = nn.MSELoss(reduction="mean")


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


class Net(nn.Module):
    def __init__(self, trial, input_dim: int, output_dim: int):
        super(Net, self).__init__()
        self.layers = []
        self.dropouts = []
        self.input_dim = input_dim
        self.output_dim = output_dim

        # We optimize the number of layers, hidden units in each layer and dropouts.
        n_layers = trial.suggest_int("n_layers", 1, 8)
        dropout = trial.suggest_float("dropout", 0.2, 0.5)
        for i in range(n_layers):
            output_head = trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True)
            self.layers.append(nn.Linear(input_dim, output_head))
            self.dropouts.append(nn.Dropout(dropout))
            input_dim = output_head

        self.layers.append(nn.Linear(input_dim, output_dim))

        # Assigning the layers as class variables (PyTorch requirement).
        # Parameters of a layer are returned when calling model.parameters(),
        # only if the layer is a class variable. Thus, assigning as class
        # variable is necessary to make the layer parameters trainable.
        for idx, layer in enumerate(self.layers):
            setattr(self, "fc{}".format(idx), layer)

        # Assigning the dropouts as class variables (PyTorch requirement), for
        # the same reason as above.
        for idx, dropout in enumerate(self.dropouts):
            setattr(self, "drop{}".format(idx), dropout)

    def forward(self, data):
        for layer, dropout in zip(self.layers, self.dropouts):
            data = F.relu(layer(data))
            data = dropout(data)
        return nn.Linear(self.input_dim, self.output_dim)
        # return F.linear(data)


class LightningNet(pl.LightningModule):
    def __init__(self, trial):
        super(LightningNet, self).__init__()
        self.model = Net(trial, input_dim=input_shape, output_dim=output_shape)

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_nb):
        data, target = batch
        output = self.forward(data)
        return {"loss": F.nll_loss(output, target)}

    def validation_step(self, batch, batch_nb):
        data, target = batch
        output = self.forward(data)
        loss = mse_loss(output, target)
        # pred = output.argmax(dim=1, keepdim=True)
        # accuracy = pred.eq(target.view_as(pred)).float().mean()
        return {"mse_loss": loss}

    def validation_epoch_end(self, outputs):
        ave_mse_loss = np.mean(x["mse_loss"] for x in outputs) / len(outputs)
        # Pass the ave_mse_loss to the `DictLogger` via the `'log'` key.
        return {"log": {"mse_loss": ave_mse_loss}}

    def configure_optimizers(self):
        return Adam(self.model.parameters())

    def train_dataloader(self):
        return DataLoader(
            dataset,
            batch_size=BATCHSIZE,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset,
            batch_size=BATCHSIZE,
            shuffle=False,
        )


def objective(trial):
    # Filenames for each trial must be made unique in order to access each checkpoint.
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, "trial_{}".format(trial.number), "{epoch}"),
        monitor="val_acc",
    )

    # The default logger in PyTorch Lightning writes to event files to be consumed by
    # TensorBoard. We don't use any logger here as it requires us to implement several abstract
    # methods. Instead we setup a simple callback, that saves metrics from each validation step.
    metrics_callback = MetricsCallback()
    trainer = pl.Trainer(
        logger=False,
        limit_val_batches=PERCENT_VALID_EXAMPLES,
        checkpoint_callback=checkpoint_callback,
        max_epochs=EPOCHS,
        gpus=1 if torch.cuda.is_available() else None,
        callbacks=[
            metrics_callback,
            PyTorchLightningPruningCallback(trial, monitor="val_acc"),
        ],
    )

    model = LightningNet(trial)
    trainer.fit(model)

    return metrics_callback.metrics[-1]["val_acc"].item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
        "trials at the early stages of training.",
    )
    args = parser.parse_args()

    pruner = (
        optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    )

    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=100, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
