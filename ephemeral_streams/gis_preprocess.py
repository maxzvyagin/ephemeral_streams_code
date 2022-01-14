### Module in order to preprocess the satellite images and ready them for input into deep learning models
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
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pdb
from sklearn.preprocessing import MinMaxScaler


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
    mask = mask.astype(float)
    # mask[mask == 1] = 255
    return mask


def mask_from_output(model_output):
    """Given model output, softmax probability for 2 classes, generate a mask corresponding to segmentation"""
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
    """split a given 3d array into 4 equal quarters"""
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

def process_image(image_array, image_type):
    if image_type == "full_channel":
        return image_array
    elif image_type == "rgb":
        return image_array[:3]
    elif image_type == "ir":
        return image_array[3]
    elif image_type == "hsv":
        new = rgb2hsv(np.moveaxis(image_array[:3], 0, -1))
        new = np.moveaxis(new, -1, 0)
        return new
    elif image_type == "hsv_with_ir":
        new = rgb2hsv(np.moveaxis(image_array[:3], 0, -1))
        new = np.moveaxis(new, -1, 0)
        all_channels = np.concatenate((new, np.expand_dims(image_array[3], 0)), axis=0)
        return all_channels
    elif image_type == "veg_index":
        # red channel
        r = np.expand_dims(image_array[0], 0)
        # infrared channel
        i = np.expand_dims(image_array[3], 0)
        veg = numpy_msavi(r, i)
        return veg
    else:
        raise ValueError("Could not find image type {}".format(image_type))


def get_windows(img_f, mask, large_image=False, unlabelled=False, num=500, get_max=True, rand=False, image_type="full",
                only_mask=True):
    if large_image:
        window_size = 512
    else:
        window_size = 256

    samples = []

    with rasterio.open(img_f) as src:
        full_image = src.read()

    max_x = mask.shape[0] // window_size
    max_y = mask.shape[1] // window_size

    print(mask.shape)
    print(full_image.shape)

    for i in tqdm(range(max_x)):
        for j in range(max_y):
            mask_window = mask[i * window_size:(i + 1) * window_size, j * window_size:(j + 1) * window_size]
            if (only_mask and 1 in mask_window) or not only_mask:
                # need to grab all channels
                # image_window = full_image[:, i * window_size:(i + 1) * window_size, j * window_size:(j + 1) * window_size]
                image_window = full_image[:, i * window_size:(i + 1) * window_size,
                               j * window_size:(j + 1) * window_size]
                # need to check if image is alright - some portions are masked but not image actually present
                if np.std(image_window) != 0:
                    image_window = process_image(image_window, image_type=image_type)
                    window = (torch.from_numpy(image_window).half(), torch.from_numpy(mask_window).int())
                    samples.append(window)

    return samples


# given red and infrared reflectance values, calculate the vegetation index (desert version from Yuki's paper)
def msavi(red, infrared):
    return (2 * infrared) + 1 - math.sqrt((2 * infrared + 1) ** 2 - (8 * (infrared - red))) / 2


numpy_msavi = np.vectorize(msavi)


def generate_rotated_samples(samples):
    """For each sample, generate 3 90 degree rotations - data augmentation method"""
    rotated_samples = []
    for sample in tqdm(samples):
        rotated_image, rotated_mask = sample
        rotated_image = rotated_image.squeeze()
        rotated_mask = rotated_mask.squeeze()
        sample_image = rotated_image
        sample_mask = rotated_mask
        if len(rotated_image.shape) > 2:
            raise NotImplementedError
        for _ in range(3):
            rotated_image = np.rot90(rotated_image).copy()
            rotated_mask = np.rot90(rotated_mask).copy()
            # generate tensors
            saved_image = torch.from_numpy(np.expand_dims(rotated_image, 0)).half()
            saved_mask = torch.from_numpy(rotated_mask).int()
            window = (saved_image, saved_mask)
            rotated_samples.append(window)
        # flip up down and left right
        for flip_function in [np.flipud, np.fliplr]:
            flip_image = flip_function(sample_image).copy()
            flip_mask = flip_function(sample_mask).copy()
            saved_image = torch.from_numpy(np.expand_dims(flip_image, 0)).half()
            saved_mask = torch.from_numpy(flip_mask).int()
            window = (saved_image, saved_mask)
            rotated_samples.append(window)

    return rotated_samples


def get_samples(img_and_shps, image_type, large_image, only_mask=False):
    samples = []
    for pair in img_and_shps:
        print("Processing file {}....".format(pair[0]))
        mask = mask_from_shp(pair[0], pair[1])
        # trying out this swap
        # mask = np.swapaxes(mask, 0, 1)
        windows = get_windows(pair[0], mask, large_image, image_type=image_type, only_mask=only_mask)
        samples.extend(windows)
    return samples


def pt_gis_train_test_split(img_and_shps=None, image_type="rgb", large_image=False, theta=True):
    """ Return PT GIS Datasets with Train Test Split"""

    if not img_and_shps:
        img_and_shps = [("/scratch/mzvyagin/Ephemeral_Channels/Imagery/vhr_2012_refl.img",
                         "/scratch/mzvyagin/Ephemeral_Channels/Reference/reference_2012_merge.shp"),]
                        # ("/scratch/mzvyagin/Ephemeral_Channels/Imagery/vhr_2014_refl.img",
                        #  "/scratch/mzvyagin/Ephemeral_Channels/Reference/reference_2014_merge.shp")]

    # first see if we have a cached object, will use this name to cache if doesn't exist
    name = "/tmp/mzvyagin/"
    name += "gis_data"
    name += image_type
    if large_image:
        name += "large_image"
    name += "PTdataset.pkl"
    if path.exists(name):
        try:
            cache_object = open(name, "rb")
            train, val, test = pickle.load(cache_object)
            print("WARNING: Loaded from pickle object at {}...".format(name))
            return PT_GISDataset(train), PT_GISDataset(val), PT_GISDataset(test)
        except:
            print("ERROR: could not load from cache file. Please try removing " + name + " and try again.")
            sys.exit()

    # no cache object was found, so we generate from scratch - training first
    with_mask = get_samples(img_and_shps, image_type=image_type, large_image=large_image, only_mask=True)
    rotated_samples = generate_rotated_samples(with_mask)
    with_mask = with_mask + rotated_samples
    random.shuffle(with_mask)

    # then test
    img_and_shps = [("/scratch/mzvyagin/Ephemeral_Channels/Imagery/vhr_2014_refl.img",
                         "/scratch/mzvyagin/Ephemeral_Channels/Reference/reference_2014_merge.shp")]
    with_mask_test = get_samples(img_and_shps, image_type=image_type, large_image=large_image, only_mask=True)
    rotated_samples_test = generate_rotated_samples(with_mask_test)
    with_mask_test = with_mask_test + rotated_samples_test

    # train, test = train_test_split(with_mask, train_size=0.8, shuffle=True, random_state=0)
    train = with_mask
    val, test = train_test_split(with_mask_test, train_size=0.1, shuffle=True, random_state=0)
    cache_object = open(name, "wb")
    pickle.dump((train, val, test), cache_object)
    return PT_GISDataset(train), PT_GISDataset(val), PT_GISDataset(test)


class PT_GISDataset(Dataset):
    """Generates a dataset for Pytorch of image and labelled mask."""

    # need to be given a list of tuple consisting of filepaths, (img, shp) to get pairs of windows for training
    def __init__(self, data_list):
        # can be initialized from a list of samples instead of from files
        self.samples = data_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        x, y = self.samples[index]
        return x.float(), y.float()


def pt_to_tf(x):
    """ Converts a pytorch tensor to a tensorflow tensor and returns it"""
    n = x.numpy()
    n = np.swapaxes(n, 0, -1)
    t = tf.convert_to_tensor(n)
    return t
