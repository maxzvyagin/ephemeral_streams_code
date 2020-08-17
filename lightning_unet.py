import preprocess
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import argparse
from pytorch_lightning.logging.neptune import NeptuneLogger
import time
import statistics

# Defining Environment Variables - defaults defined here and edited using command line args
MAX_EPOCHS = 25
LR = 1e-3
BATCHSIZE = 64
INPUT_CHANNELS = 4
OUTPUT_CHANNELS = 1
NUM_GPUS = 1
IMAGE_TYPE = "full_channel"
REP = 32
LARGE_IMAGE=False


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
        all_data = preprocess.GISDataset(self.file_pairs, IMAGE_TYPE, LARGE_IMAGE)
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
        return DataLoader(self.test_set, batch_size=BATCHSIZE, num_workers=0)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=LR)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        start = time.time()
        x = train_batch['image']
        if IMAGE_TYPE == "veg_index":
            x = x.unsqueeze(1)
        y = train_batch['mask'].unsqueeze(1)
        # x, y = train_batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        end = time.time()
        time_spent = end - start
        logs = {'train_loss': loss, 'batch_time': time_spent}
        return {'loss': loss, 'batch_time': time_spent, 'log': logs}

    def training_epoch_end(self, outputs):
        times = []
        for x in outputs:
            times.append(x['batch_time'])
        avg_time_per_batch = statistics.mean(times)
        #avg_time_per_batch = torch.stack([x['batch_time'] for x in outputs]).mean()
        tensorboard_logs = {'avg_time_per_batch': avg_time_per_batch}
        return {'avg_time_per_batch': avg_time_per_batch, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x = batch['image']
        if IMAGE_TYPE == "veg_index":
            x = x.unsqueeze(1)
        y = batch['mask'].unsqueeze(1)
        # x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([torch.Tensor(x['test_loss']) for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    def validation_step(self, val_batch, batch_idx):
        x = val_batch['image']
        if IMAGE_TYPE == "veg_index":
            x = x.unsqueeze(1)
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
    # parsing arguments and parameters for the model
    parser = argparse.ArgumentParser(description="Input Parameters for UNet Model Training")
    parser.add_argument("-e", "--experiment_name", required=True)
    parser.add_argument("-f", "--machine", required=True)
    parser.add_argument("-i", "--image_type", help="Specify if using all 4 channels, just RGB, IR, HSV, etc.",
                        required=True)
    parser.add_argument("-b", "--batchsize")
    parser.add_argument("-g", "--gpus", required=True, help="Comma separated list of selected GPUs, string format.")
    parser.add_argument("-m", "--max_epochs")
    parser.add_argument("-l", "--lr")
    parser.add_argument("-t", "--tags", help="Comma separated list of tags for Neptune, string format.")
    parser.add_argument("-r", "--representation", help="Enter 16 if 16 bit representation is desired. Else leave off.")
    parser.add_argument("-s", "--big_image", help="Enter True if 512 image is desired, instead of 256.")
    args = parser.parse_args()
    if args.image_type:
        IMAGE_TYPE = args.image_type
    if args.batchsize:
        BATCHSIZE = int(args.batchsize)
    if args.gpus:
        gpus = args.gpus.split(",")
        gpus = list(map(int, gpus))
        NUM_GPUS = len(gpus)
    if args.max_epochs:
        MAX_EPOCHS = int(args.max_epochs)
    if args.lr:
        LR = float(args.lr)
    if args.tags:
        tags = args.tags.split(",")
    if args.representation:
        r = int(args.representation)
        if r == 32:
            pass
        else:
            REP = 16
            print("NOTE: Using 16 bit integer representation.")
    if args.big_image:
        LARGE_IMAGE=True
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
    if args.machine == "nucleus":
        f = [("/vol/ml/EphemeralStreamData/Ephemeral_Channels/Imagery/vhr_2012_refl.img"
              , "/vol/ml/EphemeralStreamData/Ephemeral_Channels/Reference/reference_2012_merge.shp"),
             ("/vol/ml/EphemeralStreamData/Ephemeral_Channels/Imagery/vhr_2014_refl.img",
              "/vol/ml/EphemeralStreamData/Ephemeral_Channels/Reference/reference_2014_merge.shp")]
    elif args.machine == "lambda":
        f = [("/scratch/mzvyagin/Ephemeral_Channels/Imagery/vhr_2012_refl.img",
              "/scratch/mzvyagin/Ephemeral_Channels/Reference/reference_2012_merge.shp"),
             ("/scratch/mzvyagin/Ephemeral_Channels/Imagery/vhr_2014_refl.img",
              "/scratch/mzvyagin/Ephemeral_Channels/Reference/reference_2014_merge.shp")]
    else:
        f = [("/lus/iota-fs0/projects/CVD_Research/Ephemeral_Channels/Imagery/vhr_2012_refl.img",
              "/lus/iota-fs0/projects/CVD_Research/Ephemeral_Channels/Reference/reference_2012_merge.shp"),
             ("/lus/iota-fs0/projects/CVD_Research/Ephemeral_Channels/Imagery/vhr_2014_refl.img",
              "/lus/iota-fs0/projects/CVD_Research/Ephemeral_Channels/Reference/reference_2014_merge.shp")]

    nep = NeptuneLogger(api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5"
                                "lcHR1bmUuYWkiLCJhcGlfa2V5IjoiOGE5NDI0YTktNmE2ZC00ZWZjLTlkMjAtNjNmMTIwM2Q2ZTQzIn0=",
                        project_name="maxzvyagin/GIS", experiment_name=args.experiment_name, close_after_fit=False,
                        params={"batch_size": BATCHSIZE, "num_gpus": NUM_GPUS, "learning_rate": LR,
                                "image_type": IMAGE_TYPE, "max_epochs": MAX_EPOCHS, "precision": REP}, tags=tags)
    model = LitUNet(f, INPUT_CHANNELS, OUTPUT_CHANNELS)
    if REP == 16:
        trainer = pl.Trainer(gpus=gpus, max_epochs=MAX_EPOCHS, logger=nep, precision=16)
    else:
        trainer = pl.Trainer(gpus=gpus, max_epochs=MAX_EPOCHS, logger=nep)
    start = time.time()
    trainer.fit(model)
    end = time.time()
    nep.log_metric("clock_time(s)", end - start)
    # run the test set
    trainer.test(model)
    torch.save(model.state_dict(), "/tmp/latest_model.pkl")
    nep.log_artifact("/tmp/latest_model.pkl")
