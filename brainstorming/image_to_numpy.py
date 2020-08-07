### utility script to open a GIS file, convert to a numpy array, and then save it to a numpy file

import rasterio
import numpy as np

with rasterio.open("/Users/mzvyagin/Documents/GISProject/nucleus_data/Ephemeral_Channels/Imagery/vhr_2012_refl.img") as src:
    print(src.count)
    print(src.width)
    print(src.height)
    print("---------------------")
    image = np.empty((src.count, src.width, src.height),dtype="float32")
    i = 0
    for ji, window in src.block_windows():
        r = src.read(window=window)
        #print(window)
        #print(r.shape)
        #print(image[:, window.col_off:window.col_off+r.shape[1], window.row_off:window.row_off+r.shape[2]].index)
        try:
            image[:, window.col_off:window.col_off+r.shape[1], window.row_off:window.row_off+r.shape[2]] = r
        except Exception as e:
            i += 1
            print(i)
            print(e)
            pass
    # now save the numpy array to a numpy file
    np.save("twelve_image.npy", image)