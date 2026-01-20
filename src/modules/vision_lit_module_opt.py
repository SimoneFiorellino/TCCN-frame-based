from pathlib import Path

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from src.modules.model import Decoder


class LitModule(LightningModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        encoder: str,
        batch_size: int = 256,
        num_workers: int = 0,
        pin_memory: bool = False,
        n_anchors: int = 500,
        n_classes: int = 20,
        dataset: str = "cifar100",
        relative: bool = True,
    ):
        super().__init__()

        # save hyperparameters
        self.save_hyperparameters()

        # loss
        self.loss = nn.CrossEntropyLoss()

        self.labels = "labels" if dataset != "cifar20" else "coarse_labels"
        dataset = dataset if dataset != "cifar20" else "cifar100"

        print(f"labels: {self.labels}")

        LATENTS_DIR = Path(f"./data/latents/{dataset}")

        print(f"Dataset: {dataset}.")
        print("Check labels!\n")

        print(f"Loading latents from {LATENTS_DIR}...")
        self.trainset = torch.load(LATENTS_DIR / "train" / f"{encoder}.pt")
        self.valset = torch.load(LATENTS_DIR / "val" / f"{encoder}.pt")
        self.testset = torch.load(LATENTS_DIR / "test" / f"{encoder}.pt")

        print("Creating decoder network.\n")
        # save network architecture
        self.net = Decoder(
            in_features=n_anchors
            if self.hparams.relative
            else self.trainset["absolute"].shape[1],
            out_features=n_anchors
            if self.hparams.relative
            else int(0.5 * self.trainset["absolute"].shape[1]),
            n_classes=n_classes,
        )

        print("Creating metric objects.\n")
        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=n_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=n_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=n_classes)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

        # save optimizer and scheduler
        self.optimizer = optimizer
        self.scheduler = scheduler

    def forward(self, x):
        return self.net(x)

    def affine_transform(self, x, anchs):
        """
        random sample a matrix A and apply it to x, from a gaussian distribution with mean 1 and std 0.1
        random sample a vector b and apply it to x, from a gaussian distribution with mean 0 and std 0.1

        x: tensor of shape: (batch, features)

        return tensor of shape: (batch, features)
        """

        A = torch.normal(mean=1, std=2, size=(x.shape[1], x.shape[1])).to(x.device)
        B = torch.normal(mean=0, std=0.5, size=(x.shape[1],)).to(x.device)

        new_x = torch.matmul(x, A) + B
        new_anchs = torch.matmul(anchs, A) + B

        return new_x, new_anchs

    def model_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)  # [:, self.idx])
        loss = self.loss(y_hat, y)
        return loss, y_hat, y

    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self.model_step(batch, batch_idx)
        self.train_loss(loss)
        self.train_acc(y_hat, y)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self.model_step(batch, batch_idx)
        self.val_loss(loss)
        self.val_acc(y_hat, y)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self.model_step(batch, batch_idx)
        self.test_loss(loss)
        self.test_acc(y_hat, y)
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def train_dataloader(self):
        return DataLoader(
            TensorDataset(self.trainset["absolute"], self.trainset[self.labels]),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            TensorDataset(self.valset["absolute"], self.valset[self.labels]),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            TensorDataset(self.testset["absolute"], self.testset[self.labels]),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
