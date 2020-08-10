### Module in order to preprocess the satellite images and ready them for input into PyTorch
### Max Zvyagin

import rasterio
import rasterio.features
import geopandas as gpd
import torch
from torch.utils.data import Dataset
import math


def mask_from_shp(img_f, shp_f):
    # this function will also perform the reprojection onto the image file that we're masking (CRS adjustment)
    # read in the shp file
    shp = gpd.read_file(shp_f)
    img = rasterio.open(img_f)
    # get the crs from the image file
    new_crs = str(img.crs).lower()
    # perform the reprojection
    shp_reproject = shp.to_crs({'init': new_crs})
    # now that the shapes are lined up, get the mask from the .shp geometry
    geometry = shp_reproject['geometry']
    mask = rasterio.features.geometry_mask(geometry, img.shape, img.transform, all_touched=False, invert=True)
    return mask

def split(array):
    # split a given 3d array into 4 equal chunks
    if len(array.shape) == 2:
        mid_x = int(array.shape[0]/2)
        mid_y = int(array.shape[1]/2)
        first = array[:mid_x, :mid_y]
        second = array[mid_x:, :mid_y]
        third = array[:mid_x, mid_y:]
        fourth = array[mid_x:, mid_y:]
    else:
        mid_x = int(array.shape[1]/2)
        mid_y = int(array.shape[2]/2)
        first = array[:, :mid_x, :mid_y]
        second = array[:, mid_x:, :mid_y]
        third = array[:, :mid_x, mid_y:]
        fourth = array[:, mid_x:, mid_y:]
    chunks = [first, second, third, fourth]
    return chunks


# given the name of an image file and the corresponding .shp array mask, outputs an array of image windows and mask windows
def get_windows(img_f, mask):
    samples = []
    with rasterio.open(img_f) as src:
        for ji, window in src.block_windows():
            # get the window from the mask
            mask_check = mask[window.row_off:window.row_off+window.height, window.col_off:window.col_off+window.width]
            if True in mask_check:
                # need to split into tiles
                r = src.read(window=window)
                r_chunks = split(r)
                mask_chunks = split(mask_check)
                for i in range(4):
                    samples.append((torch.from_numpy(r_chunks[i]), torch.from_numpy(mask_chunks[i])))
                # also can probably convert to tensors here as well
               # samples.append((torch.from_numpy(r),torch.from_numpy(mask_check)))
            else:
                pass
    return samples

# given the name of an image file and the corresponding .shp array mask, outputs an array of calculated vegetation index values and mask
def get_vegetation_index_windows(img_f, mask):
    samples = []
    with rasterio.open(img_f) as src:
        for ji, window in src.block_windows():
            # get the window from the mask
            mask_check = mask[window.row_off:window.row_off+window.height, window.col_off:window.col_off+window.width]
            if True in mask_check:
                # need to split into tiles
                r = src.read(2, window=window)
                i = src.read(3, window=window)
                msavi = msavi(r, i)
                chunks = split(msavi)
                mask_chunks = split(mask_check)
                for i in range(4):
                    samples.append((torch.from_numpy(r_chunks[i]), torch.from_numpy(mask_chunks[i])))
                # also can probably convert to tensors here as well
               # samples.append((torch.from_numpy(r),torch.from_numpy(mask_check)))
            else:
                pass
    return samples

# given red and infrared reflectance values, calculate the vegetation index
def msavi(red, infrared):
    return ((2*infrared)+1-math.sqrt((2*infrared+1)**2-(8*(infrared-red)))/2)

class GISDataset(Dataset):
    # need to be given a list of tuple consisting of filepaths, (img, shp) to get pairs of windows for training
    def __init__(self, img_and_shps):
        self.samples = []
        for pair in img_and_shps:
            # process each pair and generate the windows
            mask = mask_from_shp(pair[0], pair[1])
            windows = get_windows(pair[0], mask)
            self.samples.extend(windows)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        res = {}
        pulled_sample = self.samples[index]
        res['image'] = pulled_sample[0]
        res['mask'] = pulled_sample[1]
        return res
