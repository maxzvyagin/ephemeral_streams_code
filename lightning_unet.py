import preprocess
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import argparse
from torch.nn import functional as F
from pytorch_lightning.loggers import TensorBoardLogger, NeptuneLogger

# Defining Global Variables
MAX_EPOCHS=25
LR=1e-3
BATCHSIZE=64
INPUT_CHANNELS=4
OUTPUT_CHANNELS=1
NUM_GPUS=1

class LitUNet(pl.LightningModule):

    def __init__(self, file_pairs, input_num=4, output_num=1, initial_feat=32, trained=False):
        super().__init__()
        self.model = model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=input_num,
                                            out_channels=output_num,
                                            init_features=initial_feat, pretrained=trained)
        self.file_pairs = file_pairs
        self.criterion = torch.nn.MSELoss(reduction="mean")

    def forward(self, x):
        return self.model(x)

    def prepare_data(self):
        all_data = preprocess.GISDataset(self.file_pairs)
        # calculate the splits
        total = len(all_data)
        train = int(total * .7)
        val = int(total * .15)
        if train + (val * 2) != total:
            diff = total - train - (val * 2)
            train += diff
        self.train_set, self.validate_set, self.test_set = torch.utils.data.random_split(all_data, [train, val, val])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=64, num_workers=10)

    def val_dataloader(self):
        return DataLoader(self.validate_set, batch_size=64, num_workers=10)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=64, num_workers=10)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x = train_batch['image']
        y = train_batch['mask'].unsqueeze(1)
        # x, y = train_batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)

        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def test_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['mask'].unsqueeze(1)
        # x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    def validation_step(self, val_batch, batch_idx):
        x = val_batch['image']
        y = val_batch['mask'].unsqueeze(1)
        # x, y = val_batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        # called at the end of the validation epoch
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

# Argument Parsing and Run Training
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters for UNet Model Training")
    ### 
    parser.add_argument("--image-type", help="Specify if using all 4 channels, just RGB, IR, HSV, etc.")
    parser.add