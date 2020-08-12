import preprocess
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import argparse
from pytorch_lightning.logging.neptune import NeptuneLogger

# Defining Global Variables
MAX_EPOCHS = 25
LR = 1e-3
BATCHSIZE = 64
INPUT_CHANNELS = 4
OUTPUT_CHANNELS = 1
NUM_GPUS = 1
IMAGE_TYPE = "full_channel"


class LitUNet(pl.LightningModule):

    def __init__(self, file_pairs, input_num=4, output_num=1, initial_feat=32, trained=False):
        super().__init__()
        self.model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=input_num,
                                    out_channels=output_num,
                                    init_features=initial_feat, pretrained=trained)
        self.file_pairs = file_pairs
        self.criterion = torch.nn.MSELoss(reduction="mean")
        # initialize dataset variables
        self.train_set = None
        self.validate_set = None
        self.test_set = None

    def forward(self, x):
        return self.model(x)

    def prepare_data(self):
        all_data = preprocess.GISDataset(self.file_pairs, IMAGE_TYPE)
        # calculate the splits
        total = len(all_data)
        train = int(total * .7)
        val = int(total * .15)
        if train + (val * 2) != total:
            diff = total - train - (val * 2)
            train += diff
        # get splits and store in object
        self.train_set, self.validate_set, self.test_set = torch.utils.data.random_split(all_data, [train, val, val])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=BATCHSIZE, num_workers=10)

    def val_dataloader(self):
        return DataLoader(self.validate_set, batch_size=BATCHSIZE, num_workers=10)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=BATCHSIZE, num_workers=10)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=LR)
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


if __name__ == "__main__":
    print("in main")
    # parsing arguments and parameters for the model
    parser = argparse.ArgumentParser(description="Input Parameters for UNet Model Training")
    parser.add_argument("-e", "--experiment_name", required=True)
    parser.add_argument("-f", "--training_files", required=True, action="append", nargs="+", type=str)
    parser.add_argument("-i", "--image_type", help="Specify if using all 4 channels, just RGB, IR, HSV, etc.",
                        required=True)
    parser.add_argument("-b", "--batchsize")
    parser.add_argument("-n", "--num_gpus")
    parser.add_argument("-m", "--max_epochs")
    parser.add_argument("-l", "--lr")
    args = parser.parse_args()
    if args.image_type:
        IMAGE_TYPE = args.image_type
    if args.batchsize:
        BATCHSIZE = args.batchsize
    if args.num_gpus:
        NUM_GPUS = args.num_gpus
    if args.max_epochs:
        MAX_EPOCHS = args.max_epochs
    if args.lr:
        LR = args.lr
    # need to figure out how many input channels we have
    if IMAGE_TYPE == "full_channel":
        INPUT_CHANNELS = 4
    elif IMAGE_TYPE == "rgb":
        INPUT_CHANNELS = 3
    elif IMAGE_TYPE == "ir":
        INPUT_CHANNELS = 1
    elif IMAGE_TYPE == "hsv":
        INPUT_CHANNELS = 3
    elif IMAGE_TYPE == "hsv_with_ir":
        INPUT_CHANNELS = 4
    elif IMAGE_TYPE == "veg_index":
        INPUT_CHANNELS = 1
    else:
        print("WARNING: no image type match, defaulting to RGB+IR")
        INPUT_CHANNELS = 4
        IMAGE_TYPE = "full_channel"
    # initialize a model
    f = []
    for x in args.training_files:
        latest = tuple(x)
        f.append(latest)
    nep = NeptuneLogger(api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5"
                                   "lcHR1bmUuYWkiLCJhcGlfa2V5IjoiOGE5NDI0YTktNmE2ZC00ZWZjLTlkMjAtNjNmMTIwM2Q2ZTQzIn0=",
                           project_name="GIS/segmentation", experiment_name=args.experiment_name,
                           params={"batch_size": BATCHSIZE, "num_gpus": NUM_GPUS, "learning_rate": LR,
                                   "image_type": IMAGE_TYPE, "max_epochs": MAX_EPOCHS})
    model = LitUNet(f, INPUT_CHANNELS, OUTPUT_CHANNELS)
    trainer = pl.Trainer(gpus=NUM_GPUS, auto_select_gpus=True, max_epochs=MAX_EPOCHS, logger=nep)
    trainer.fit(model)
    trainer.test(model)