#!/bin/bash
python lightning_unet.py -f nucleus -e nucleus_fullchannel -i full_channel -g 3 -l 5e-6 -m 25 -t nucleus,itype,precision
python lightning_unet.py -f nucleus -e nucleus_fullchannel_16bit -i full_channel -g 3 -l 5e-6 -m 25 -t nucleus,itype,precision -r 16
python lightning_unet.py -f nucleus -e nucleus_rgb -i rgb -g 3 -l 5e-6 -m 25 -t nucleus,itype
python lightning_unet.py -f nucleus -e nucleus_ir -i ir -g 3 -l 5e-6 -m 25 -t nucleus,itype
python lightning_unet.py -f nucleus -e nucleus_hsv -i hsv -g 3 -l 5e-6 -m 25 -t nucleus,itype
python lightning_unet.py -f nucleus -e nucleus_hsv_with_ir -i hsv_with_ir -g 3 -l 5e-6 -m 25 -t nucleus,itype
python lightning_unet.py -f nucleus -e nucleus_veg_index -i veg_index -g 3 -l 5e-6 -m 25 -t nucleus,itype


#python lightning_unet.py -f nucleus -e nucleus_fullchannel -i full_channel -g 3 -l 5e-6 -m 5 -t nucleus,itype,precision
# python lightning_pannet.py -f nucleus -e nucleus_fullchannel -i full_channel -g 3 -l 5e-6 -m 25 -t nucleus,itype,precision
# python lightning_unet_selftraining.py -f nucleus -e nucleus_selftrain -i full_channel -g 3 -l 5e-6 -m 2 -t selftraintest
# python lightning_unet.py -f nucleus -e nucleus_highlr -i hsv_with_ir -g 3 -m 25 -t lr_test