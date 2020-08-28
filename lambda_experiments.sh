#!/bin/bash
python lightning_unet.py -f lambda -e lambda_fullchannel -i full_channel -g 7 -l 5e-6 -m 25 -t lambda,itype,precision
python lightning_unet.py -f lambda -e lambda_fullchannel_16bit -i full_channel -g 7 -l 5e-6 -m 25 -t lambda,itype,precision -r 16
python lightning_unet.py -f lambda -e lambda_rgb -i rgb -g 7 -l 5e-6 -m 25 -t lambda,itype
python lightning_unet.py -f lambda -e lambda_ir -i ir -g 7 -l 5e-6 -m 25 -t lambda,itype
python lightning_unet.py -f lambda -e lambda_hsv -i hsv -g 7 -l 5e-6 -m 25 -t lambda,itype
python lightning_unet.py -f lambda -e lambda_hsv_with_ir -i hsv_with_ir -g 7 -l 5e-6 -m 25 -t lambda,itype
python lightning_unet.py -f lambda -e lambda_veg_index -i veg_index -g 7 -l 5e-6 -m 25 -t lambda,itype


# python lightning_unet.py -f lambda -e lambda_fullchannel -i full_channel -g 7 -l 5e-6 -m 100 -t lambda,itype,size -s True
# python lightning_unet.py -f lambda -e lambda_fullchannel -i full_channel -g 7 -l 5e-6 -m 25 -t lambda,profiling
# python lightning_unet_selftraining.py -f lambda -e lambda_selftrain -i full_channel -g 3 -l 5e-6 -m 1 -t selftraintest,random_sampling
# python lightning_unet.py -f lambda -e lambda_fullchannel -i full_channel -g 7 -l 1e-8 -m 50 -t lambda,2channel
# python lightning_unet.py -f lambda -e lambda_hsv_with_ir -i hsv_with_ir -g 7 -l 5e-7 -m 25 -t lambda,hsv,lr