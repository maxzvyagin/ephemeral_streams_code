### script to generate testing of hyperparameter tuning
import numpy as np

f = open('hyperparameter_search.sh', "w")
f.write('#!/bin/bash\n')
x = 0
for lr in np.linspace(.00000001, .1, 8):
    for wd in np.linspace(.00000001, .1, 8):
        for d in np.arange(0, 1, .1):
            x += 1
            line = 'python lightning_unet.py -c -t gridsearch -g 7 -f lambda -i full_channel -m 50 -d '+str(d)+' -l '+str(lr)+' -w '+str(wd)+' -e lambda_weight_lr_dropout_exp'+str(x)
            line += "\n"
            f.write(line)
f.close()