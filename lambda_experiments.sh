#!/bin/bash
python lightning_unet.py -f lambda -e lambda_fullchannel -i full_channel -g 7 -l 5e-6 -m 100 -t lambda,itype,precision
python lightning_unet.py -f lambda -e lambda_fullchannel_16bit -i full_channel -g 7 -l 5e-6 -m 100 -t lambda,itype,precision -r 16
python lightning_unet.py -f lambda -e lambda_rgb -i rgb -g 7 -l 5e-6 -m 100 -t lambda,itype
python lightning_unet.py -f lambda -e lambda_ir -i ir -g 7 -l 5e-6 -m 100 -t lambda,itype
python lightning_unet.py -f lambda -e lambda_hsv -i hsv -g 7 -l 5e-6 -m 100 -t lambda,itype
python lightning_unet.py -f lambda -e lambda_hsv_with_ir -i hsv_with_ir -g 7 -l 5e-6 -m 100 -t lambda,itype
python lightning_unet.py -f lambda -e lambda_veg_index -i veg_index -g 7 -l 5e-6 -m 100 -t lambda,itype