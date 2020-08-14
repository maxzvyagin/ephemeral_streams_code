!/bin/bash
python lightning_unet.py -f lambda -e lambda_rgb -i rgb -g 7 -l 5e-6 -m 100 -t lambda,itype
python lightning_unet.py -f lambda -e lambda_ir -i ir -g 7 -l 5e-6 -m 100 -t lambda,itype
python lightning_unet.py -f lambda -e lambda_hsv -i hsv -g 7 -l 5e-6 -m 100 -t lambda,itype
python lightning_unet.py -f lambda -e lambda_hsv_with_ir -i hsv_with_ir -g 7 -l 5e-6 -m 100 -t lambda,itype
python lightning_unet.py -f lambda -e lambda_veg_index -i veg_index -g 7 -l 5e-6 -m 100 -t lambda,itype