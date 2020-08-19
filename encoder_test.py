

import os

encoders = ['resnet152', 'dpn107', 'vgg19', 'se_resnet101', 'densenet161', 'inceptionv4', 'efficientnet-b6',
            'xception', 'timm-efficientnet-b7']

for enc in encoders:
    os.system('python lightning_unet.py -f lambda -e lambda_fullchannel -i full_channel -g 7 -l 5e-6 -m 25 '
              '-t lambda,encoder,'+enc+" -a "+enc)
    