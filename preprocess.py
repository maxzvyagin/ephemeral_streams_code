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
import pickle
from os import path
import sys
import random


def mask_from_shp(img_f, shp_f):
    # this function will also perform the reprojection onto the image file that we're masking (CRS adjustment)
    # read in the shp file
    shp = gpd.read_file(shp_f)
    img = rasterio.open(img_f)
    # get the crs from the image file
    new_crs = str(img.crs).lower()
    # perform the reprojection
    shp_reproject = shp.to_crs(new_crs)
    # now that the shapes are lined up, get the mask from the .shp geometry
    geometry = shp_reproject['geometry']
    mask = rasterio.features.geometry_mask(geometry, img.shape, img.transform, all_touched=False, invert=True)
    return mask


def mask_from_output(model_output):
    # given model output, softmax probability for 2 classes, generate a mask corresponding to segmentation
    # get final shape of output
    final_shape = model_output.shape[-2:]
    print(final_shape)
    result = []
    channel_one = torch.flatten(model_output[0])
    channel_two = torch.flatten(model_output[1])
    for i in range(len(torch.flatten(model_output[0]))):
        if channel_one[i] >= channel_two[i]:
            result.append(0)
        else:
            result.append(1)
    result = torch.reshape(torch.FloatTensor(result), final_shape).unsqueeze(0)
    return result




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
# @lru_cache(maxsize=2)
def get_windows(img_f, mask, large_image=False, unlabelled=False, num=500, get_max=True, rand=False):
    samples = []
    with rasterio.open(img_f) as src:
        image = src.block_windows()
        if rand:
            random.shuffle(list(image))
        for ji, window in image:
            if len(samples) >= num and not get_max:
                return samples
            # get the window from the mask
            mask_check = mask[window.row_off:window.row_off + window.height,
                         window.col_off:window.col_off + window.width]
            if True in mask_check:
                if not unlabelled:
                    r = src.read(window=window)
                    if large_image:
                        samples.append((torch.from_numpy(r).float(), torch.from_numpy(mask_check).float()))
                    else:
                        # need to split into tiles
                        r_chunks = split(r)
                        mask_chunks = split(mask_check)
                        for i in range(4):
                            samples.append(
                                (torch.from_numpy(r_chunks[i]).float(), torch.from_numpy(mask_chunks[i]).float()))
            else:
                if unlabelled:
                    r = src.read(window=window)
                    if large_image:
                        chunk_size = r.shape
                        if chunk_size[-1] == 512 and chunk_size[-2] == 512:
                            samples.append(torch.from_numpy(r).float())
                    else:
                        # need to split into tiles
                        r_chunks = split(r)
                        for i in range(4):
                            chunk_size = r_chunks[i].shape
                            if chunk_size[-1] == 256 and chunk_size[-2] == 256:
                                samples.append(torch.from_numpy(r_chunks[i]).float())
    return samples


# return 3 channel image of rgb reflectance values
# @lru_cache(maxsize=2)
def get_rgb_windows(img_f, mask, large_image=False, unlabelled=False, num=500, get_max=True, rand=False):
    samples = []
    with rasterio.open(img_f) as src:
        if rand:
            image = random.shuffle(list(src.block_windows()))
        else:
            image = src.block_windows()
        for ji, window in image:
            if len(samples) >= num and not get_max:
                return samples
            # get the window from the mask
            mask_check = mask[window.row_off:window.row_off + window.height,
                         window.col_off:window.col_off + window.width]
            if True in mask_check:
                if not unlabelled:
                    r = src.read(window=window)
                    if large_image:
                        samples.append((torch.from_numpy(r[:3]).float(), torch.from_numpy(mask_check).float()))
                    else:
                        # need to split into tiles
                        r_chunks = split(r)
                        mask_chunks = split(mask_check)
                        for i in range(4):
                            samples.append(
                                (torch.from_numpy(r_chunks[i][:3]).float(), torch.from_numpy(mask_chunks[i]).float()))
            else:
                if unlabelled:
                    r = src.read(window=window)
                    if large_image:
                        chunk_size = r.shape
                        if chunk_size[-1] == 512 and chunk_size[-2] == 512:
                            samples.append(torch.from_numpy(r[:3]).float())
                    else:
                        # need to split into tiles
                        r_chunks = split(r)
                        for i in range(4):
                            chunk_size = r_chunks[i].shape
                            if chunk_size[-1] == 256 and chunk_size[-2] == 256:
                                samples.append(torch.from_numpy(r_chunks[i][:3]).float())
    return samples


# return single channel image of solely infrared reflectance values
# @lru_cache(maxsize=2)
def get_ir_windows(img_f, mask, large_image=False, unlabelled=False, num=500, get_max=True, rand=False):
    samples = []
    with rasterio.open(img_f) as src:
        if rand:
            image = random.shuffle(list(src.block_windows()))
        else:
            image = src.block_windows()
        for ji, window in image:
            if len(samples) >= num and not get_max:
                return samples
            # get the window from the mask
            mask_check = mask[window.row_off:window.row_off + window.height,
                         window.col_off:window.col_off + window.width]
            if True in mask_check:
                if not unlabelled:
                    r = src.read(window=window)
                    if large_image:
                        samples.append((torch.from_numpy(r[3]).float(), torch.from_numpy(mask_check).float()))
                    else:
                        # need to split into tiles
                        r_chunks = split(r)
                        mask_chunks = split(mask_check)
                        for i in range(4):
                            samples.append(
                                (torch.from_numpy(r_chunks[i][3]).float(), torch.from_numpy(mask_chunks[i]).float()))
            else:
                if unlabelled:
                    r = src.read(window=window)
                    if large_image:
                        chunk_size = r.shape
                        if chunk_size[-1] == 512 and chunk_size[-2] == 512:
                            samples.append(torch.from_numpy(r[3]).float())
                    else:
                        # need to split into tiles
                        r_chunks = split(r)
                        for i in range(4):
                            chunk_size = r_chunks[i].shape
                            if chunk_size[-1] == 256 and chunk_size[-2] == 256:
                                samples.append(torch.from_numpy(r_chunks[i][3]).float())
    return samples


# return 3 channels of rgb converted to hsv
# @lru_cache(maxsize=2)
def get_hsv_windows(img_f, mask, large_image=False, unlabelled=False, num=500, get_max=True, rand=False):
    samples = []
    with rasterio.open(img_f) as src:
        if rand:
            image = random.shuffle(list(src.block_windows()))
        else:
            image = src.block_windows()
        for ji, window in image:
            if len(samples) >= num and not get_max:
                return samples
            # get the window from the mask
            mask_check = mask[window.row_off:window.row_off + window.height,
                         window.col_off:window.col_off + window.width]
            if True in mask_check:
                if not unlabelled:
                    r = src.read(window=window)
                    if large_image:
                        new_val = rgb2hsv(np.moveaxis(r[:3], 0, -1))
                        new_val = np.moveaxis(new_val, -1, 0)
                        samples.append((torch.from_numpy(new_val).float(), torch.from_numpy(mask_check).float()))
                    else:
                        # need to split into tiles
                        r_chunks = split(r)
                        mask_chunks = split(mask_check)
                        for i in range(4):
                            new_val = rgb2hsv(np.moveaxis(r_chunks[i][:3], 0, -1))
                            new_val = np.moveaxis(new_val, -1, 0)
                            samples.append(
                                (torch.from_numpy(new_val).float(), torch.from_numpy(mask_chunks[i]).float()))
            else:
                if unlabelled:
                    r = src.read(window=window)
                    if large_image:
                        chunk_size = r.shape
                        if chunk_size[-1] == 512 and chunk_size[-2] == 512:
                            new_val = rgb2hsv(np.moveaxis(r[:3], 0, -1))
                            new_val = np.moveaxis(new_val, -1, 0)
                            samples.append(torch.from_numpy(new_val).float())
                    else:
                        # need to split into tiles
                        r_chunks = split(r)
                        for i in range(4):
                            chunk_size = r_chunks[i].shape
                            if chunk_size[-1] == 256 and chunk_size[-2] == 256:
                                new_val = rgb2hsv(np.moveaxis(r_chunks[i][:3], 0, -1))
                                new_val = np.moveaxis(new_val, -1, 0)
                                samples.append(torch.from_numpy(new_val).float())
    return samples


# return rgb converted to hsv in addition to infrared channel
def get_hsv_with_ir_windows(img_f, mask, large_image=False, unlabelled=False, num=500, get_max=True, rand=False):
    samples = []
    with rasterio.open(img_f) as src:
        if rand:
            image = random.shuffle(list(src.block_windows()))
        else:
            image = src.block_windows()
        for ji, window in image:
            if len(samples) >= num and not get_max:
                return samples
            # get the window from the mask
            mask_check = mask[window.row_off:window.row_off + window.height,
                         window.col_off:window.col_off + window.width]
            if True in mask_check:
                if not unlabelled:
                    r = src.read(window=window)
                    if large_image:
                        new_val = rgb2hsv(np.moveaxis(r[:3], 0, -1))
                        new_val = np.moveaxis(new_val, -1, 0)
                        all_channels = np.concatenate((new_val, np.expand_dims(r[3], 0)), axis=0)
                        samples.append((torch.from_numpy(all_channels).float(), torch.from_numpy(mask_check).float()))
                    else:
                        # need to split into tiles
                        r_chunks = split(r)
                        mask_chunks = split(mask_check)
                        for i in range(4):
                            new_val = rgb2hsv(np.moveaxis(r_chunks[i][:3], 0, -1))
                            new_val = np.moveaxis(new_val, -1, 0)
                            all_channels = np.concatenate((new_val, np.expand_dims(r_chunks[i][3], 0)), axis=0)
                            samples.append(
                                (torch.from_numpy(all_channels).float(), torch.from_numpy(mask_chunks[i]).float()))
        else:
            if unlabelled:
                r = src.read(window=window)
                if large_image:
                    chunk_size = r.shape
                    if chunk_size[-1] == 512 and chunk_size[-2] == 512:
                        new_val = rgb2hsv(np.moveaxis(r[:3], 0, -1))
                        new_val = np.moveaxis(new_val, -1, 0)
                        all_channels = np.concatenate((new_val, np.expand_dims(r[3], 0)), axis=0)
                        samples.append(torch.from_numpy(all_channels).float())
                else:
                    # need to split into tiles
                    r_chunks = split(r)
                    for i in range(4):
                        chunk_size = r_chunks[i].shape
                        if chunk_size[-1] == 256 and chunk_size[-2] == 256:
                            new_val = rgb2hsv(np.moveaxis(r_chunks[i][:3], 0, -1))
                            new_val = np.moveaxis(new_val, -1, 0)
                            all_channels = np.concatenate((new_val, np.expand_dims(r_chunks[i][3], 0)), axis=0)
                            samples.append(torch.from_numpy(all_channels).float())
    return samples


# given the name of an image file and the corresponding .shp array mask, outputs an array of calculated vegetation index values and mask
def get_vegetation_index_windows(img_f, mask, large_image=False, unlabelled=False, num=500, get_max=True, rand=False):
    samples = []
    with rasterio.open(img_f) as src:
        if rand:
            image = random.shuffle(list(src.block_windows()))
        else:
            image = src.block_windows()
        for ji, window in image:
            if len(samples) >= num and not get_max:
                return samples
            # get the window from the mask
            mask_check = mask[window.row_off:window.row_off + window.height,
                         window.col_off:window.col_off + window.width]
            if True in mask_check:
                if not unlabelled:
                    r = src.read(2, window=window)
                    i = src.read(3, window=window)
                    veg = numpy_msavi(r, i)
                    if large_image:
                        samples.append((torch.from_numpy(veg).float(), torch.from_numpy(mask_check).float()))
                    else:
                        chunks = split(veg)
                        mask_chunks = split(mask_check)
                        # the split function return 4 separate quadrants from the original window
                        for i in range(4):
                            samples.append((torch.from_numpy(chunks[i]).float(),
                                            torch.from_numpy(mask_chunks[i]).float()))
            else:
                if unlabelled:
                    r = src.read(2, window=window)
                    i = src.read(3, window=window)
                    veg = numpy_msavi(r, i)
                    if large_image:
                        chunk_size = veg.shape
                        if chunk_size[-1] == 512 and chunk_size[-2] == 512:
                            samples.append(torch.from_numpy(veg).float())
                    else:
                        r_chunks = split(veg)
                        # the split function return 4 separate quadrants from the original window
                        for i in range(4):
                            chunk_size = r_chunks[i].shape
                            if chunk_size[-1] == 256 and chunk_size[-2] == 256:
                                samples.append(torch.from_numpy(chunks[i]).float())
                pass
    return samples


# given red and infrared reflectance values, calculate the vegetation index (desert version from Yuki's paper)
def msavi(red, infrared):
    return (2 * infrared) + 1 - math.sqrt((2 * infrared + 1) ** 2 - (8 * (infrared - red))) / 2


numpy_msavi = np.vectorize(msavi)


class GISDataset(Dataset):
    """Generates a dataset for Pytorch of image and labelled mask."""
    # need to be given a list of tuple consisting of filepaths, (img, shp) to get pairs of windows for training
    def __init__(self, img_and_shps, image_type, large_image=False, data_list=None, iota=False):
        # can be initialized from a list of samples instead of from files
        if data_list:
            self.samples = data_list
            self.image_type = image_type
        else:
            self.samples = []
            self.image_type = image_type
            for pair in img_and_shps:
                # check if there is a cached object available
                if iota:
                    name = "/raid/scratch/mzvyagin/"
                else:
                    name = "/tmp/"
                name += pair[0].split("/")[-1]
                name += image_type
                if large_image:
                    name += "large_image"
                name += "dataset.pkl"
                if path.exists(name):
                    try:
                        cache_object = open(name, "rb")
                        windows = pickle.load(cache_object)
                    except:
                        print("ERROR: could not load from cache file. Please try removing " + name + " and try again.")
                        sys.exit()
                # process each pair and generate the windows
                else:
                    mask = mask_from_shp(pair[0], pair[1])
                    if image_type == "full_channel":
                        windows = get_windows(pair[0], mask, large_image)
                    elif image_type == "rgb":
                        windows = get_rgb_windows(pair[0], mask, large_image)
                    elif image_type == "ir":
                        windows = get_ir_windows(pair[0], mask, large_image)
                    elif image_type == "hsv":
                        windows = get_hsv_windows(pair[0], mask, large_image)
                    elif image_type == "hsv_with_ir":
                        windows = get_hsv_with_ir_windows(pair[0], mask, large_image)
                    elif image_type == "veg_index":
                        windows = get_vegetation_index_windows(pair[0], mask, large_image)
                    else:
                        print("WARNING: no image type match, defaulting to RGB+IR")
                        windows = get_windows(pair[0], mask, large_image)
                    # cache the windows
                    cache_object = open(name, "wb+")
                    pickle.dump(windows, cache_object)
                self.samples.extend(windows)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        res = {}
        pulled_sample = self.samples[index]
        res['image'] = pulled_sample[0]
        res['mask'] = pulled_sample[1]
        return res


class UnlabelledGISDataset(Dataset):
    """ Used for sampling for unsupervised learning purposes."""
    def __init__(self, img_and_shps, image_type, large_image=False, num_images=500):
        self.samples = []
        self.image_type = image_type
        for pair in img_and_shps:
            # check if there is a cached object available
            name = "/tmp/"
            name += pair[0].split("/")[-1]
            name += image_type
            if large_image:
                name += "large_image"
            name += "unlabelled_dataset.pkl"
            if path.exists(name):
                try:
                    cache_object = open(name, "rb")
                    windows = pickle.load(cache_object)
                except:
                    print("ERROR: could not load from cache file. Please try removing " + name + " and try again.")
                    sys.exit()
            # process each pair and generate the windows
            else:
                mask = mask_from_shp(pair[0], pair[1])
                if image_type == "full_channel":
                    windows = get_windows(pair[0], mask, large_image, unlabelled=True, num=num_images, get_max=False,
                                          rand=True)
                elif image_type == "rgb":
                    windows = get_rgb_windows(pair[0], mask, large_image, unlabelled=True, num=num_images, get_max=False,
                                              rand=True)
                elif image_type == "ir":
                    windows = get_ir_windows(pair[0], mask, large_image, unlabelled=True, num=num_images, get_max=False,
                                             rand=True)
                elif image_type == "hsv":
                    windows = get_hsv_windows(pair[0], mask, large_image, unlabelled=True, num=num_images, get_max=False)
                elif image_type == "hsv_with_ir":
                    windows = get_hsv_with_ir_windows(pair[0], mask, large_image, unlabelled=True, num=num_images,
                                                      get_max=False, rand=True)
                elif image_type == "veg_index":
                    windows = get_vegetation_index_windows(pair[0], mask, large_image, unlabelled=True, num=num_images,
                                                           get_max=False, rand=True)
                else:
                    print("WARNING: no image type match, defaulting to RGB+IR")
                    windows = get_windows(pair[0], mask, large_image, unlabelled=True, num=num_images, rand=True)
                # cache the windows
                cache_object = open(name, "wb+")
                pickle.dump(windows, cache_object)
            self.samples.extend(windows)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
