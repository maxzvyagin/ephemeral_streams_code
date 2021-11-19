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
import math

from ephemeral_streams import pt_gis_train_test_split
from torch.utils.data import DataLoader

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import wandb

from pytorch_lightning.loggers import WandbLogger

import matplotlib.pyplot as plt

import torchmetrics

from transformers import SegformerModel, SegformerConfig, SegformerForSemanticSegmentation
from cv2 import resize

import pdb


# def custom_transform(img):
#     return torchvision.transforms.ToTensor(np.array(img))


### definition of PyTorch Lightning module in order to run everything
def iou_loss(iou_value):
    return -torch.log(iou_value)


def iou(pred, target):
    intersection = torch.logical_and(pred, target)
    union = torch.logical_or(pred, target)
    value = torch.sum(intersection) / torch.sum(union)
    value.requires_grad = True
    return value


class PyTorch_UNet(pl.LightningModule):
    def __init__(self, config, classes, in_channels=1, model_type="deeplabv3", image_type="veg_index"):
        super(PyTorch_UNet, self).__init__()
        self.config = config
        # sigmoid is part of BCE with logits loss
        # self.model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        #                             in_channels=in_channels, out_channels=classes, init_features=32, pretrained=True)
        # self.model = smp.MAnet(encoder_name="resnet34", encoder_weights=None, in_channels=in_channels, classes=classes)
        self.model = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None, in_channels=in_channels,
                                      classes=classes, decoder_attention_type="scse")
        # configuration = SegformerConfig(image_size=256, num_channels=1, num_labels=1, hidden_dropout_prob=0.5, attention_probs_dropout_prob=0.5,
        #                                 classifier_dropout_prob=0.5)
        # self.model = SegformerForSemanticSegmentation(configuration)
        # self.criterion = nn.BCEWithLogitsLoss()
        # self.criterion = smp.losses.DiceLoss(mode="binary")
        self.criterion = iou_loss
        self.test_loss = None
        self.test_accuracy = None
        self.test_iou = None
        self.learning_rate = 0.0
        self.accuracy = torchmetrics.Accuracy()
        self.f1 = torchmetrics.F1()
        # self.iou = torchmetrics.IoU(num_classes=1)
        self.train_set, self.valid_set, self.test_set = pt_gis_train_test_split(image_type=image_type)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=int(self.config['batch_size']), num_workers=5)

    def val_dataloader(self):
        return DataLoader(self.valid_set, batch_size=int(self.config['batch_size']), num_workers=5)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=int(self.config['batch_size']), num_workers=5)

    def configure_optimizers(self):
        # auto finding learning rate
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # wandb.log({'learning_rate': self.learning_rate})
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'])
        # optimizer = torch.optim.RMSprop(self.parameters(), lr=self.config['learning_rate'])
        return optimizer

    def forward(self, x):
        # out = self.model(x).logits
        # need to scale up resolution
        # out = torchvision.transforms.functional.resize(out, [256, 256])
        # return out
        return self.model(x)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        return {'forward': self.forward(x), 'expected': y}

    def training_step_end(self, outputs):
        # only use when  on dp
        output = outputs['forward'].squeeze(1)
        # loss = self.criterion(output, outputs['expected'])
        f1 = self.f1(output, outputs['expected'].int())
        # now compute iou
        output = torch.nn.Sigmoid()(output).int()
        accuracy = self.accuracy(output, outputs['expected'].int())
        iou_value = iou(output, outputs['expected'].int())
        loss = iou_loss(iou_value)
        # iou_value = iou_value.detach().cpu()
        # accuracy = accuracy.detach().cpu()
        # iou = self.iou(output, outputs['expected'].int()).detach().cpu()
        # recall = self.recall_metric(output, outputs['expected'].int()).detach().cpu()
        # precision = self.precision_metric(output, outputs['expected'].int()).detach().cpu()
        logs = {'train_loss': loss, 'train_accuracy': accuracy, 'train_iou': iou_value,
                'train_f1': f1}
        self.log("training", logs)
        return {'loss': loss, 'logs': logs, 'train_accuracy': accuracy, 'train_iou': iou_value, 'train_f1': f1}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        return {'forward': self.forward(x), 'expected': y}

    def validation_step_end(self, outputs):
        output = outputs['forward'].squeeze(1)
        # loss = self.criterion(output, outputs['expected'])
        f1 = self.f1(output, outputs['expected'].int())
        # now compute iou
        output = torch.nn.Sigmoid()(output).int()
        accuracy = self.accuracy(output, outputs['expected'].int()).detach().cpu()
        iou_value = iou(output, outputs['expected'].int())
        loss = iou_loss(iou_value)
        iou_value = iou_value.detach().cpu()
        # iou = self.iou(output, outputs['expected'].int()).detach().cpu()
        # recall = self.recall_metric(output, outputs['expected'].int()).detach().cpu()
        # precision = self.precision_metric(output, outputs['expected'].int()).detach().cpu()
        logs = {'val_loss': loss.detach().cpu(), 'val_accuracy': accuracy, 'val_iou': iou_value,
                'val_f1': f1}
        self.log("validation", logs)
        return {'loss': loss, 'logs': logs, 'val_accuracy': accuracy, 'val_iou': iou_value, 'val_f1': f1}

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        return {'forward': self.forward(x), 'expected': y}

    def test_step_end(self, outputs):
        output = outputs['forward'].squeeze(1)
        # loss = self.criterion(output, outputs['expected'])
        f1 = self.f1(output, outputs['expected'].int())
        # now compute iou
        output = torch.nn.Sigmoid()(output).int()
        accuracy = self.accuracy(output, outputs['expected'].int()).detach().cpu()
        iou_value = iou(output, outputs['expected'].int())
        loss = iou_loss(iou_value)
        iou_value = iou_value.detach().cpu()
        # iou = self.iou(output, outputs['expected'].int()).detach().cpu()
        # recall = self.recall_metric(output, outputs['expected'].int()).detach().cpu()
        # precision = self.precision_metric(output, outputs['expected'].int()).detach().cpu()
        logs = {'test_loss': loss.detach().cpu(), 'test_accuracy': accuracy, 'test_iou': iou_value,
                'test_f1': f1}
        self.log("testing", logs)
        return {'loss': loss, 'logs': logs, 'test_accuracy': accuracy, 'test_iou': iou_value, 'test_f1': f1}

    def test_epoch_end(self, outputs):
        loss = []
        accuracy = []
        iou_values = []
        for x in outputs:
            loss.append(float(x['test_loss']))
            accuracy.append(float(x['test_accuracy']))
            iou_values.append(float(x['test_iou']))
        avg_loss = statistics.mean(loss)
        self.test_loss = avg_loss

        avg_accuracy = statistics.mean(accuracy)
        self.test_accuracy = avg_accuracy

        avg_iou = statistics.mean(iou_values)
        self.test_iou = avg_iou

        tensorboard_logs = {'test_loss': avg_loss, 'test_accuracy': avg_accuracy, 'test_iou': avg_iou}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs, 'avg_test_accuracy': avg_accuracy,
                'avg_test_iou': avg_iou}


def generate_test_segmentations(model):
    model.model.eval()
    model.model.cuda()
    fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(10, 4))
    ax[0][0].set_title("Prediction")
    ax[0][1].set_title("Real")
    ax[0][2].set_title("Image")
    with torch.no_grad():
        selected_indices = []
        total_num = len(model.test_set)
        selected_indices.append(3)
        selected_indices.append(total_num // 2)
        selected_indices.append(total_num - 3)
        for n, i in enumerate([0, 75, 150]):
            # run through the model
            x, y = model.test_set[i]
            out = model.forward(x.unsqueeze(0).cuda())
            out = torch.nn.Sigmoid()(out)
            out = np.rint(out.cpu().numpy().reshape(256, 256))
            # generate the images
            ax[n][0].imshow(out, cmap="cividis")
            ax[n][1].imshow(y.cpu().numpy(), cmap="cividis")
            ax[n][2].imshow(x.cpu().squeeze().numpy(), cmap="binary")
    filename = '/tmp/mzvyagin/segmentation.png'
    plt.savefig(filename, dpi=300)
    wandb.log({"segmentation_maps": wandb.Image(filename)})


def segmentation_pt_objective(config):
    torch.manual_seed(0)
    model = PyTorch_UNet(config, classes=1, in_channels=1)
    wandb_logger = WandbLogger()
    trainer = pl.Trainer(max_epochs=config['epochs'], gpus=1, auto_select_gpus=True, logger=wandb_logger)
    # callbacks=[EarlyStopping(monitor="val_accuracy")])
    trainer.tune(model)
    trainer.fit(model)
    trainer.test(model)
    generate_test_segmentations(model)
    return model.test_accuracy, model.model


### two different objective functions, one for cityscapes and one for GIS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', default=8)
    args = parser.parse_args()
    test_config = {'batch_size': args.batch_size, 'learning_rate': 0.001, 'epochs': 50}
    wandb.init(project='ephemeral_streams', entity='mzvyagin', config=test_config)
    acc, model = segmentation_pt_objective(test_config)
    torch.save(model, "/tmp/mzvyagin/ephemeral_streams_model.pkl")
    # torch.save(model, "initial_model.pkl")
