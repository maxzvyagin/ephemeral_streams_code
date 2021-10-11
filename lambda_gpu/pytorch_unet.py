### PyTorch UNet
import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
import statistics
import numpy as np
import os
import sys
import argparse
import segmentation_models_pytorch as smp

from gis_preprocess import pt_gis_train_test_split
from torch.utils.data import DataLoader

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import wandb

from pytorch_lightning.loggers import WandbLogger


# def custom_transform(img):
#     return torchvision.transforms.ToTensor(np.array(img))


### definition of PyTorch Lightning module in order to run everything
class PyTorch_UNet(pl.LightningModule):
    def __init__(self, config, classes, in_channels=3, model_type="deeplabv3", image_type="hsv_with_ir"):
        super(PyTorch_UNet, self).__init__()
        self.config = config
        # sigmoid is part of BCE with logits loss
        # self.model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        #                             in_channels=in_channels, out_channels=classes, init_features=32, pretrained=True)
        self.model = smp.DeepLabV3(encoder_weights=None, in_channels=in_channels, classes=classes)
        self.criterion = nn.BCEWithLogitsLoss()
        self.test_loss = None
        self.test_accuracy = None
        self.test_iou = None
        self.accuracy = pl.metrics.Accuracy()
        self.train_set, self.valid_set, self.test_set = pt_gis_train_test_split(image_type=image_type)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=int(self.config['batch_size']), num_workers=5)

    def val_dataloader(self):
        return DataLoader(self.valid_set, batch_size=int(self.config['batch_size']), num_workers=5)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=int(self.config['batch_size']), num_workers=5)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'])
        return optimizer

    def forward(self, x):
        return self.model(x)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        return {'forward': self.forward(x), 'expected': y}

    def training_step_end(self, outputs):
        # only use when  on dp
        loss = self.criterion(outputs['forward'].squeeze(1), outputs['expected'])
        logs = {'train_loss': loss}
        self.log(logs)
        return {'loss': loss, 'logs': logs}

    def val_step(self, val_batch, batch_idx):
        x, y = val_batch
        return {'forward': self.forward(x), 'expected': y}

    def val_step_end(self, outputs):
        # only use when  on dp
        loss = self.criterion(outputs['forward'].squeeze(1), outputs['expected'])
        logs = {'val_loss': loss}
        self.log(logs)
        return {'loss': loss, 'logs': logs}

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        return {'forward': self.forward(x), 'expected': y}

    def test_step_end(self, outputs):
        loss = self.criterion(outputs['forward'].squeeze(1), outputs['expected'])
        accuracy = self.accuracy(outputs['forward'].squeeze(1), outputs['expected'])
        logs = {'test_loss': loss, 'test_accuracy': accuracy}
        self.log(logs)
        return {'test_loss': loss, 'logs': logs, 'test_accuracy': accuracy}

    def test_epoch_end(self, outputs):
        loss = []
        for x in outputs:
            loss.append(float(x['test_loss']))
        avg_loss = statistics.mean(loss)
        tensorboard_logs = {'test_loss': avg_loss}
        self.test_loss = avg_loss
        accuracy = []
        for x in outputs:
            accuracy.append(float(x['test_accuracy']))
        avg_accuracy = statistics.mean(accuracy)
        self.test_accuracy = avg_accuracy
        # iou = []
        # for x in outputs:
        #     iou.append(float(x['test_iou']))
        # avg_iou = statistics.mean(iou)
        # self.test_iou = avg_iou
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs, 'avg_test_accuracy': avg_accuracy}


def segmentation_pt_objective(config):
    torch.manual_seed(0)
    model = PyTorch_UNet(config, classes=1, in_channels=3)
    trainer = pl.Trainer(max_epochs=config['epochs'], gpus=1, auto_select_gpus=True,
                         callbacks=[EarlyStopping(monitor="val_loss")], wandb_logger=WandbLogger())
    trainer.fit(model)
    trainer.test(model)
    return model.test_accuracy, model.model


### two different objective functions, one for cityscapes and one for GIS

if __name__ == "__main__":
    wandb.init(project='ephemeral_streams', entity='mzvyagin')
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', default=64)
    args = parser.parse_args()
    test_config = {'batch_size': args.batch_size, 'learning_rate': .00001, 'epochs': 10}
    acc, model = segmentation_pt_objective(test_config)
    torch.save(model, "/tmp/mzvyagin/ephemeral_streams_model.pkl")
    # torch.save(model, "initial_model.pkl")
