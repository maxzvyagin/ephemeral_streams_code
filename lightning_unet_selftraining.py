### Semi Supervised Self Training Version of lightning_unet.py ###

import preprocess
import pytorch_lightning as pl
import torch
import argparse
from pytorch_lightning.logging.neptune import NeptuneLogger
import time
import lightning_unet

# Defining Environment Variables - defaults defined here and edited using command line args
MAX_EPOCHS = 25
LR = 1e-3
BATCHSIZE = 64
INPUT_CHANNELS = 4
OUTPUT_CHANNELS = 1
NUM_GPUS = 1
IMAGE_TYPE = "full_channel"
REP = 32
LARGE_IMAGE = False
ENCODER = None


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
    parser.add_argument("-a", "--encoder", help="Specify an encoder for unet if desired, default is blank."
                                                "See Github for SMP for options.")
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
    if args.encoder:
        ENCODER = args.encoder
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
        f = [("/home/Imagery/vhr_2012_refl.img",
              "/home/Reference/reference_2012_merge.shp"),
             ("/home/Imagery/vhr_2014_refl.img",
              "/home/Reference/reference_2014_merge.shp")]

    nep = NeptuneLogger(api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5"
                                "lcHR1bmUuYWkiLCJhcGlfa2V5IjoiOGE5NDI0YTktNmE2ZC00ZWZjLTlkMjAtNjNmMTIwM2Q2ZTQzIn0=",
                        project_name="maxzvyagin/GIS", experiment_name=args.experiment_name, close_after_fit=False,
                        params={"batch_size": BATCHSIZE, "num_gpus": NUM_GPUS, "learning_rate": LR,
                                "image_type": IMAGE_TYPE, "max_epochs": MAX_EPOCHS, "precision": REP}, tags=tags)
    ### set up the initial model and train on the labelled images
    model = lightning_unet.LitUNet(f, INPUT_CHANNELS, OUTPUT_CHANNELS)
    if REP == 16:
        trainer = pl.Trainer(gpus=gpus, max_epochs=MAX_EPOCHS, logger=nep, profiler=True, precision=16)
    else:
        trainer = pl.Trainer(gpus=gpus, max_epochs=MAX_EPOCHS, profiler=True, logger=nep)
    start = time.time()
    trainer.fit(model)
    end = time.time()
    nep.log_metric("clock_time(s)", end - start)
    # perform self training for 10 iterations
    model.first_run_flag = False
    print("\n\nBeginning self training...\n\n")
    unlabelled = preprocess.UnlabelledGISDataset(f, IMAGE_TYPE, LARGE_IMAGE)
    for i in range(10):
        print("Iteration {}...".format(i))
        all_data = model.train_set
        new_data = []
        model.eval()
        for x in range(len(unlabelled)):
            # need to unsqueeze in order to fix batch issue
            res = model(unlabelled[x].unsqueeze(0))
            res = res.squeeze(0)
            res = res.squeeze(0)
            try:
                print(res.size())
            except:
                print(res.shape)
            new_data.append((unlabelled[x], res))
            # new_data.append({'image': unlabelled[x], 'mask': res})
            #all_data.append({'image': unlabelled[x], 'mask': res}
        all_data = all_data + preprocess.GISDataset(None, IMAGE_TYPE, list=new_data)
        model.train_set = all_data
        # train the model again using the augemented data set
        model.train()
        trainer.fit(model)
    trainer.test(model)
    torch.save(model.state_dict(), "/tmp/latest_model.pkl")
    nep.log_artifact("/tmp/latest_model.pkl")
