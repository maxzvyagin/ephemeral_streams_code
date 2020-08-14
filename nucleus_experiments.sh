#!/bin/bash
python lightning_unet.py -f nucleus -e nucleus_ir -i ir -g 3 -l 5e-6 -m 100 -t nucleus,itype
python lightning_unet.py -f nucleus -e nucleus_hsv -i hsv -g 3 -l 5e-6 -m 100 -t nucleus,itype
python lightning_unet.py -f nucleus -e nucleus_hsv_with_ir -i hsv_with_ir -g 3 -l 5e-6 -m 100 -t nucleus,itype
python lightning_unet.py -f nucleus -e nucleus_veg_index -i veg_index -g 3 -l 5e-6 -m 100 -t nucleus,itype