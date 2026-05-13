import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, ConfusionMatrix

from src.models.CNNMamba import DroneDetectorMamba


class DroneClassifier(pl.LightningModule):
    def __init__(self, num_classes=2, n_mels=128, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.model = DroneDetectorMamba(
            num_classes=num_classes,
            n_mels=n_mels,
            ch_in=1,
        )

        if num_classes == 2:
            weights = torch.tensor([1.0, 8.0])
        else:
            weights = torch.tensor([15.0, 15.0, 1.0])

        self.register_buffer("class_weights", weights)

        self.loss_fn = nn.CrossEntropyLoss(weight=weights)

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.conf_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.train_acc(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.val_acc(logits, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.test_acc(logits, y)
        self.test_f1(logits, y)
        self.conf_matrix.update(logits, y)

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True)
        return loss

    def on_test_epoch_end(self):
        cm = self.conf_matrix.compute()
        print(f"\nConfusion Matrix:\n{cm}")
        self.conf_matrix.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)