#### Test integration of Hyperspace package with Ray Tune in order to optimize hyperparameters in Pytorch Lightning ####
from skopt import Optimizer
from hyperspace.space import create_hyperspace
from lightning_unet import LitUNet
import torch
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger
from ray import tune
from ray.tune.suggest.skopt import SkOptSearch


# from ray.tune.suggest import Searcher
#
# ### Definition of custom search algorithm for Ray Tune
# class HyperSearch(Searcher):
#     def __init__(self, metric='avg_test_loss', mode='min', **kwargs):
#         super(HyperSearch, self).__init__(metric=metrix, mode=mode, **kwargs)
#         self.configurations = {}
#
#     def suggest(self, trial_id):
#         ### return a new set of parameters to try
#         pass
#
#     def on_trial_complete(self, trial_id, result, **kwargs):
#         ## update the optimizer with the returned value
#         pass

def train_then_test(params):
    MAX_EPOCHS = 1
    LR = params['learning_rate']
    BATCHSIZE = 64
    INPUT_CHANNELS = 4
    OUTPUT_CHANNELS = 1
    NUM_GPUS = 1
    IMAGE_TYPE = "full_channel"
    REP = 32
    # default encoder defined by smp UNet class
    ENCODER = "resnet34"
    DROPOUT = params['dropout']
    WEIGHT_DECAY = params['weight_decay']
    # ENCODER_DEPTH = params['encoder_depth']
    ENCODER_DEPTH = 5
    f = [("/lus/iota-fs0/projects/CVD_Research/mzvyagin/Ephemeral_Channels/Imagery/vhr_2012_refl.img",
          "/lus/iota-fs0/projects/CVD_Research/mzvyagin/Ephemeral_Channels/Reference/reference_2012_merge.shp"),
         ("/lus/iota-fs0/projects/CVD_Research/mzvyagin/Ephemeral_Channels/Imagery/vhr_2014_refl.img",
          "/lus/iota-fs0/projects/CVD_Research/mzvyagin/Ephemeral_Channels/Reference/reference_2014_merge.shp")]
    # f = [("/scratch/mzvyagin/Ephemeral_Channels/Imagery/vhr_2012_refl.img",
    #       "/scratch/mzvyagin/Ephemeral_Channels/Reference/reference_2012_merge.shp"),
    #      ("/scratch/mzvyagin/Ephemeral_Channels/Imagery/vhr_2014_refl.img",
    #       "/scratch/mzvyagin/Ephemeral_Channels/Reference/reference_2014_merge.shp")]
    nep = NeptuneLogger(api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5"
                                "lcHR1bmUuYWkiLCJhcGlfa2V5IjoiOGE5NDI0YTktNmE2ZC00ZWZjLTlkMjAtNjNmMTIwM2Q2ZTQzIn0=",
                        project_name="maxzvyagin/GIS", experiment_name='hyperspace', close_after_fit=False,
                        params={"batch_size": BATCHSIZE, "num_gpus": NUM_GPUS, "learning_rate": LR,
                                "image_type": IMAGE_TYPE, "max_epochs": MAX_EPOCHS, "precision": REP,
                                "dropout": DROPOUT, "weight_decay": WEIGHT_DECAY}, tags=['hyperspace'])
    model = LitUNet(f, INPUT_CHANNELS, OUTPUT_CHANNELS)
    aux = dict(dropout=DROPOUT, classes=OUTPUT_CHANNELS, activation=None)
    all_decoder_channels = [256, 128, 64, 32, 16, 8, 4, 2, 1]
    model.model = smp.Unet(ENCODER, classes=OUTPUT_CHANNELS, in_channels=INPUT_CHANNELS, aux_params=aux,
                           encoder_weights=None, encoder_depth=ENCODER_DEPTH,
                           decoder_channels=all_decoder_channels[:ENCODER_DEPTH])

    trainer = pl.Trainer(gpus=1, max_epochs=MAX_EPOCHS, logger=nep, profiler=True, precision=REP, auto_select_gpus=True)
    # begin training
    trainer.fit(model)
    # run the test set
    trainer.test(model)
    torch.save(model.state_dict(), "/tmp/latest_model.pkl")
    tune.report(avg_test_loss=model.test_loss)
    return model.test_loss


### hyperspace is a collection of scikit optimize Space objects with overlapping parameters
# generate the search space, it should output a list of the parameters to try
hyperparameters = [(0.00000001, 0.1),  # learning_rate
                   (0.0, 0.9),  # dropout
                   (0.00000001, 0.1),  # weight decay
                   (1, 6)]  # encoder depth
space = create_hyperspace(hyperparameters)

### for each space in hyperspace, we want to search the space using ray tune
for section in space:
    # create a skopt gp minimize object
    optimizer = Optimizer(section)
    search_algo = SkOptSearch(optimizer, ['learning_rate', 'dropout', 'weight_decay', 'encoder_depth'],
                              metric='avg_test_loss', mode='min')
    tune.run(train_then_test, search_alg=search_algo, num_samples=20, resources_per_trial={'gpu': 1})
