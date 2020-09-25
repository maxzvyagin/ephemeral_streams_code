### script to generate testing of hyperparameter tuning
f = open('hyperparameter_search.sh', "w")
f.write('#!/bin/bash\n')
x = 0
lr = .00000001
for l in range(8):
    lr = lr*(10**l)
    wd = .00000001
    for w in range(10):
        wd = wd*(10**w)
        for d in range(0, 1, .1):
            x += 1
            line = 'python lightning_unet.py -g 7 -f lambda -i full_channel -m 50 -d '+d+' -l '+lr+' -w '+wd+' -e lambda_weight_lr_dropout_exp'+x
            line += "\n"
            f.write(line)
f.close()