python lightning_unet.py -f iota -e iota_full_512 -i full_channel -g 1 -l 5e-6 -m 25 -t iota,itype,size -s True
python lightning_unet.py -f iota -e iota_fullchannel -i full_channel -g 0 -l 5e-6 -m 25 -t iota,itype,precision
python lightning_unet.py -f iota -e iota_fullchannel_16bit -i full_channel -g 0 -l 5e-6 -m 25 -t iota,itype,precision -r 16
python lightning_unet.py -f iota -e iota_rgb -i rgb -g 0 -l 5e-6 -m 25 -t iota,itype
python lightning_unet.py -f iota -e iota_ir -i ir -g 0 -l 5e-6 -m 25 -t iota,itype
python lightning_unet.py -f iota -e iota_hsv -i hsv -g 0 -l 5e-6 -m 25 -t iota,itype
python lightning_unet.py -f iota -e iota_hsv_with_ir -i hsv_with_ir -g 0 -l 5e-6 -m 25 -t iota,itype
python lightning_unet.py -f iota -e iota_veg_index -i veg_index -g 0 -l 5e-6 -m 25 -t iota,itype

# python lightning_unet.py -f iota -e iota_fullchannel_lr -i full_channel -g 0 -l 1e-6 -m 25 -t iota,lr
# python lightning_unet_selftraining.py -f iota -e iota_selftrain_testloss -i full_channel -g 0 -l 5e-6 -m 25 -t iota,selftrain
# python lightning_unet.py -f iota -e iota_dataaug2 -i full_channel -g 0 -l 1e-6 -m 25 -t iota,augmented_dataver2