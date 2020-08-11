### Module in order to preprocess the satellite images and ready them for input into PyTorch
### Max Zvyagin

import rasterio
import rasterio.features
import geopandas as gpd
import torch
from torch.utils.data import Dataset
import math
import numpy as np
from skimage.color import rgb2hsv
from functools import lru_cache


@lru_cache(maxsize=2)
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
        mid_x = int(array.shape[0] / 2)
        mid_y = int(array.shape[1] / 2)
        first = array[:mid_x, :mid_y]
        second = array[mid_x:, :mid_y]
        third = array[:mid_x, mid_y:]
        fourth = array[mid_x:, mid_y:]
    else:
        mid_x = int(array.shape[1] / 2)
        mid_y = int(array.shape[2] / 2)
        first = array[:, :mid_x, :mid_y]
        second = array[:, mid_x:, :mid_y]
        third = array[:, :mid_x, mid_y:]
        fourth = array[:, mid_x:, mid_y:]
    chunks = [first, second, third, fourth]
    return chunks


# given the name of an image file and the corresponding .shp array mask, outputs an array of image windows and mask windows
@lru_cache(maxsize=2)
def get_windows(img_f, mask):
    samples = []
    with rasterio.open(img_f) as src:
        for ji, window in src.block_windows():
            # get the window from the mask
            mask_check = mask[window.row_off:window.row_off + window.height,
                         window.col_off:window.col_off + window.width]
            if True in mask_check:
                # need to split into tiles
                r = src.read(window=window)
                r_chunks = split(r)
                mask_chunks = split(mask_check)
                for i in range(4):
                    samples.append((torch.from_numpy(r_chunks[i]).float(), torch.from_numpy(mask_chunks[i]).float()))
            else:
                pass
    return samples


# return 3 channel image of rgb reflectance values
@lru_cache(maxsize=2)
def get_rgb_windows(img_f, mask):
    samples = []
    with rasterio.open(img_f) as src:
        for ji, window in src.block_windows():
            # get the window from the mask
            mask_check = mask[window.row_off:window.row_off + window.height,
                         window.col_off:window.col_off + window.width]
            if True in mask_check:
                # need to split into tiles
                r = src.read(window=window)
                r_chunks = split(r)
                mask_chunks = split(mask_check)
                for i in range(4):
                    samples.append(
                        (torch.from_numpy(r_chunks[i][:3]).float(), torch.from_numpy(mask_chunks[i]).float()))
            else:
                pass
    return samples


# return single channel image of solely infrared reflectance values
@lru_cache(maxsize=2)
def get_ir_windows(img_f, mask):
    samples = []
    with rasterio.open(img_f) as src:
        for ji, window in src.block_windows():
            # get the window from the mask
            mask_check = mask[window.row_off:window.row_off + window.height,
                         window.col_off:window.col_off + window.width]
            if True in mask_check:
                # need to split into tiles
                r = src.read(window=window)
                r_chunks = split(r)
                mask_chunks = split(mask_check)
                for i in range(4):
                    samples.append(
                        (torch.from_numpy(r_chunks[i][3:]).float(), torch.from_numpy(mask_chunks[i]).float()))
            else:
                pass
    return samples


# return 3 channels of rgb converted to hsv
@lru_cache(maxsize=2)
def get_hsv_windows(img_f, mask):
    samples = []
    with rasterio.open(img_f) as src:
        for ji, window in src.block_windows():
            # get the window from the mask
            mask_check = mask[window.row_off:window.row_off + window.height,
                         window.col_off:window.col_off + window.width]
            if True in mask_check:
                # need to split into tiles
                r = src.read(window=window)
                r_chunks = split(r)
                mask_chunks = split(mask_check)
                for i in range(4):
                    new_val = rgb2hsv(r_chunks[i][:3])
                    samples.append((torch.from_numpy(new_val).float(), torch.from_numpy(mask_chunks[i]).float()))
            else:
                pass
    return samples


# return rgb converted to hsv in addition to infrared channel
@lru_cache(maxsize=2)
def get_hsv_with_ir_windows(img_f, mask):
    samples = []
    with rasterio.open(img_f) as src:
        for ji, window in src.block_windows():
            # get the window from the mask
            mask_check = mask[window.row_off:window.row_off + window.height,
                         window.col_off:window.col_off + window.width]
            if True in mask_check:
                # need to split into tiles
                r = src.read(window=window)
                r_chunks = split(r)
                mask_chunks = split(mask_check)
                for i in range(4):
                    new_val = rgb2hsv(r_chunks[i][:3])
                    all_channels = np.concatenate((new_val, r_chunks[i][3]), axis=0)
                    samples.append((torch.from_numpy(all_channels).float(), torch.from_numpy(mask_chunks[i]).float()))
            else:
                pass
    return samples


# given the name of an image file and the corresponding .shp array mask, outputs an array of calculated vegetation index values and mask
@lru_cache(maxsize=2)
def get_vegetation_index_windows(img_f, mask):
    samples = []
    with rasterio.open(img_f) as src:
        for ji, window in src.block_windows():
            # get the window from the mask
            mask_check = mask[window.row_off:window.row_off + window.height,
                         window.col_off:window.col_off + window.width]
            if True in mask_check:
                # need to split into tiles
                r = src.read(2, window=window)
                i = src.read(3, window=window)
                veg = numpy_msavi(r, i)
                chunks = split(veg)
                mask_chunks = split(mask_check)
                # the split function return 4 separate quadrants from the original window
                for i in range(4):
                    samples.append((torch.from_numpy(chunks[i]).float(), torch.from_numpy(mask_chunks[i]).float()))
                # also can probably convert to tensors here as well,
            # samples.append((torch.from_numpy(r),torch.from_numpy(mask_check)))
            else:
                pass
    return samples


# given red and infrared reflectance values, calculate the vegetation index (desert version from Yuki's paper)
def msavi(red, infrared):
    return (2 * infrared) + 1 - math.sqrt((2 * infrared + 1) ** 2 - (8 * (infrared - red))) / 2


numpy_msavi = np.vectorize(msavi)


class GISDataset(Dataset):
    # need to be given a list of tuple consisting of filepaths, (img, shp) to get pairs of windows for training
    def __init__(self, img_and_shps, image_type):
        self.samples = []
        self.image_type = image_type
        for pair in img_and_shps:
            # process each pair and generate the windows
            mask = mask_from_shp(pair[0], pair[1])
            if image_type == "full_channel":
                windows = get_windows(pair[0], mask)
            elif image_type == "rgb":
                windows = get_rgb_windows(pair[0], mask)
            elif image_type == "ir":
                windows = get_ir_windows(pair[0], mask)
            elif image_type == "hsv":
                windows = get_hsv_windows(pair[0], mask)
            elif image_type == "hsv_with_ir":
                windows = get_hsv_with_ir_windows(pair[0], mask)
            elif image_type == "veg_index":
                windows = get_vegetation_index_windows(pair[0], mask)
            else:
                print("WARNING: no image type match, defaulting to RGB+IR")
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
