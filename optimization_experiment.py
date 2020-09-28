from lightning_unet import LitUNet
import torch
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger
import time

# from pytorch_lightning.callbacks.base import Callback
# from ray.tune.integration.pytorch_lightning import TuneReportCallback
# from ray.tune.suggest import Searcher
# from ray import tune
# from ray.tune.suggest.hyperopt import HyperOptSearch

### using predefined options, perform hyperparameter optimization
#
# callback = TuneReportCallback({
#     "loss": "avg_val_loss"
# }, on="validation_end")

# class hyperspace_callback(Callback):
#     def on_validation_end(self, trainer, pl_module):
#         ### send info to hyperspace

# class HyperspaceSearcher(Searcher):
### Adapted from https://docs.ray.io/en/master/tune/tutorials/tune-pytorch-lightning.html

### USING ONLY HYPERSPACE ###
from hyperspace import hyperdrive

### we want to minimize the test loss
### definition of objective function for hyperspace
def train_then_test(params):
    MAX_EPOCHS = 25
    LR = params[0]
    BATCHSIZE = 64
    INPUT_CHANNELS = 4
    OUTPUT_CHANNELS = 1
    NUM_GPUS = 1
    IMAGE_TYPE = "full_channel"
    REP = 32
    # default encoder defined by smp UNet class
    ENCODER = "resnet34"
    DROPOUT = params[1]
    WEIGHT_DECAY = params[2]
    ENCODER_DEPTH = params[3]
    f = [("/lus/iota-fs0/projects/CVD_Research/mzvyagin/Ephemeral_Channels/Imagery/vhr_2012_refl.img",
          "/lus/iota-fs0/projects/CVD_Research/mzvyagin/Ephemeral_Channels/Reference/reference_2012_merge.shp"),
         ("/lus/iota-fs0/projects/CVD_Research/mzvyagin/Ephemeral_Channels/Imagery/vhr_2014_refl.img",
          "/lus/iota-fs0/projects/CVD_Research/mzvyagin/Ephemeral_Channels/Reference/reference_2014_merge.shp")]
    # f = [("/scratch/mzvyagin/Ephemeral_Channels/Imagery/vhr_2012_refl.img",
    #       "/scratch/mzvyagin/Ephemeral_Channels/Reference/reference_2012_merge.shp"),
    #      ("/scratch/mzvyagin/Ephemeral_Channels/Imagery/vhr_2014_refl.img",
    #       "/scratch/mzvyagin/Ephemeral_Channels/Reference/reference_2014_merge.shp")]
    # nep = NeptuneLogger(api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5"
    #                             "lcHR1bmUuYWkiLCJhcGlfa2V5IjoiOGE5NDI0YTktNmE2ZC00ZWZjLTlkMjAtNjNmMTIwM2Q2ZTQzIn0=",
    #                     project_name="maxzvyagin/GIS", experiment_name='hyperspace', close_after_fit=False,
    #                     params={"batch_size": BATCHSIZE, "num_gpus": NUM_GPUS, "learning_rate": LR,
    #                             "image_type": IMAGE_TYPE, "max_epochs": MAX_EPOCHS, "precision": REP,
    #                             "dropout": DROPOUT, "weight_decay": WEIGHT_DECAY}, tags='hyperspace')
    model = LitUNet(f, INPUT_CHANNELS, OUTPUT_CHANNELS)
    aux = dict(dropout=DROPOUT, classes=OUTPUT_CHANNELS, activation=None)
    all_decoder_channels = [256, 128, 64, 32, 16, 8, 4, 2, 1]
    model.model = smp.Unet(ENCODER, classes=OUTPUT_CHANNELS, in_channels=INPUT_CHANNELS, aux_params=aux,
                           encoder_weights=None, encoder_depth=ENCODER_DEPTH,
                           decoder_channels=all_decoder_channels[:ENCODER_DEPTH])

    trainer = pl.Trainer(gpus=1, max_epochs=MAX_EPOCHS, profiler=True, precision=REP, auto_select_gpus=True)
    # begin training
    trainer.fit(model)
    # run the test set
    trainer.test(model)
    torch.save(model.state_dict(), "/tmp/latest_model.pkl")

    return model.test_loss


if __name__ == "__main__":
    hparams = [(0.00000001, 0.1),  # learning_rate
               (0.0, 0.9),  # dropout
               (0.00000001, 0.1),  # weight decay
               (1, 6)]  # encoder depth

    hyperdrive(objective=train_then_test,
               hyperparameters=hparams,
               results_path='/home/mzvyagin/hyperspace_res',
               checkpoints_path='/home/mzvyagin/hyperspace_res',
               model="GP",
               n_iterations=50,
               verbose=True,
               random_state=0)
