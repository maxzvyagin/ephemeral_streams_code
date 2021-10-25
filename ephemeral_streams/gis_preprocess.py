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
# import imgaug as ia
# import imgaug.augmenters as iaa
# from imgaug.augmentables.segmaps import SegmentationMapsOnImage
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


# def check_if_good_sample(mask_sample):
#     # num_pos = np.count_nonzero(mask_sample == 255)
#     num_pos = np.count_nonzero(mask_sample)
#     # only collect as a sample if it makes up at least 10 percent of the image
#     if num_pos / mask_sample.size >= .05:
#         return True
#     else:
#         return False

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
        r = src.read(2, window=window)
        i = src.read(3, window=window)
        veg = numpy_msavi(r, i)
        return veg
    else:
        raise ValueError("Could not find image type {}".format(image_type))


def get_windows(img_f, mask, large_image=False, unlabelled=False, num=500, get_max=True, rand=False, image_type="full"):

    if large_image:
        window_size = 512
    else:
        window_size = 256

    samples = []

    with rasterio.open(img_f) as src:
        full_image = src.read()

    # pdb.set_trace()

    # full_image = np.swapaxes(full_image, 0, 2)
    # full_image = np.swapaxes(full_image, 0, 1)

    # scale values in place
    # for i in range(4):
    #     scaler = MinMaxScaler()
    #     full_image[i, :, :] = scaler.fit_transform(full_image[i, :, :])

    max_x = (mask.shape[0] // window_size)
    max_y = (mask.shape[1] // window_size)

    print(mask.shape)
    print(full_image.shape)

    for i in tqdm(range(max_x)):
        for j in range(max_y):
            mask_window = mask[i * window_size:(i + 1) * window_size, j * window_size:(j + 1) * window_size]
            # need to grab all channels
            image_window = full_image[:, i * window_size:(i + 1) * window_size, j * window_size:(j + 1) * window_size]
            image_window = process_image(image_window, image_type=image_type)
            window = (torch.from_numpy(image_window).half(), torch.from_numpy(mask_window).int())
            samples.append(window)

    # with rasterio.open(img_f) as src:
    #     image = src.block_windows()
    #     # image = list(image)
    #     if rand:
    #         print("WARNING: RAND NOT IMPLEMENTED")
    #         # random.shuffle(image)
    #     for ji, window in tqdm(image):
    #         if len(samples) == num and not get_max:
    #             return samples
    #         # get the window from the mask
    #         mask_check = mask[window.row_off:window.row_off + window.height,
    #                      window.col_off:window.col_off + window.width]
    #
    #         r = src.read(window=window)
    #
    #         # check if r is the correct size:
    #         if r.shape[1] == 512 and r.shape[2] == 512:
    #             # do the necessary processing of the image
    #             r = process_image(r, image_type=image_type)
    #
    #             if large_image:
    #                 if unlabelled:
    #                     samples.append(torch.from_numpy(r).half())
    #                 else:
    #                     samples.append((torch.from_numpy(r).half(), torch.from_numpy(mask_check).float()))
    #             else:
    #                 # need to split into tiles
    #                 r_chunks = split(r)
    #                 if unlabelled:
    #                     for image_chunk in r_chunks:
    #                         samples.append(
    #                             (torch.from_numpy(image_chunk).half()))
    #                 else:
    #                     mask_chunks = split(mask_check)
    #                     for image_chunk, mask_chunk in zip(r_chunks, mask_chunks):
    #                         samples.append(
    #                             (torch.from_numpy(image_chunk).half(), torch.from_numpy(mask_chunk).float()))
    return samples


# given red and infrared reflectance values, calculate the vegetation index (desert version from Yuki's paper)
def msavi(red, infrared):
    return (2 * infrared) + 1 - math.sqrt((2 * infrared + 1) ** 2 - (8 * (infrared - red))) / 2


numpy_msavi = np.vectorize(msavi)


# def augment_dataset(dataset):
#     # generate augmented samples of dataset
#     ia.seed(1)
#     # flip from left to right
#     seq = iaa.Sequential([iaa.Fliplr()])
#     augmented_samples = []
#     for sample in dataset:
#         img = sample['image'].numpy()
#         img = np.moveaxis(img, 0, -1)
#         seg = SegmentationMapsOnImage(sample['mask'].numpy().astype(bool), shape=img.shape)
#         i, s = seq(image=img, segmentation_maps=seg)
#         s = torch.FloatTensor(s.get_arr().copy())
#         i = torch.FloatTensor(np.moveaxis(i, -1, 0).copy())
#         augmented_samples.append((i, s))
#     # do the same thing but flip images upside down
#     seq = iaa.Sequential([iaa.Flipud()])
#     for sample in dataset:
#         img = sample['image'].numpy()
#         img = np.moveaxis(img, 0, -1)
#         seg = SegmentationMapsOnImage(sample['mask'].numpy().astype(bool), shape=img.shape)
#         i, s = seq(image=img, segmentation_maps=seg)
#         s = torch.FloatTensor(s.get_arr().copy())
#         i = torch.FloatTensor(np.moveaxis(i, -1, 0).copy())
#         augmented_samples.append((i, s))
#     return augmented_samples

def get_samples(img_and_shps, image_type, large_image):
    samples = []
    for pair in img_and_shps:
        print("Processing file {}....".format(pair[0]))
        mask = mask_from_shp(pair[0], pair[1])
        # trying out this swap
        mask = np.swapaxes(mask, 0, 1)
        windows = get_windows(pair[0], mask, large_image, image_type=image_type)
        samples.extend(windows)
    return samples

def pt_gis_train_test_split(img_and_shps=None, image_type="rgb", large_image=False, theta=True):
    """ Return PT GIS Datasets with Train Test Split"""

    if not img_and_shps:
        img_and_shps = [("/scratch/mzvyagin/Ephemeral_Channels/Imagery/vhr_2012_refl.img",
                         "/scratch/mzvyagin/Ephemeral_Channels/Reference/reference_2012_merge.shp"),
                        ("/scratch/mzvyagin/Ephemeral_Channels/Imagery/vhr_2014_refl.img",
                         "/scratch/mzvyagin/Ephemeral_Channels/Reference/reference_2014_merge.shp")]

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

    # no cache object was found, so we generate from scratch
    samples = get_samples(img_and_shps, image_type=image_type, large_image=large_image)

    # separating out masked sections
    with_mask = []
    for i in tqdm(samples):
        # check if 1 in mask
        if 1 in i[1]:
            with_mask.append(i)
        else:
            pass

    # need to only be grabbing parts where it's annotated, otherwise we have streams in the photo where it's not labeled
    pdb.set_trace()

    train, test = train_test_split(with_mask, train_size=0.8, shuffle=True, random_state=0)
    val, test = train_test_split(test, train_size=0.5, shuffle=True, random_state=0)
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

# def tf_gis_test_train_split(img_and_shps=None, image_type="full_channel", large_image=False, theta=True):
#     """ Returns a Tensorflow dataset of images and masks"""
#     # Default is theta file system location
#     if not img_and_shps:
#         img_and_shps = [("/scratch/mzvyagin/Ephemeral_Channels/Imagery/vhr_2012_refl.img",
#               "/scratch/mzvyagin/Ephemeral_Channels/Reference/reference_2012_merge.shp"),
#              ("/scratch/mzvyagin/Ephemeral_Channels/Imagery/vhr_2014_refl.img",
#               "/scratch/mzvyagin/Ephemeral_Channels/Reference/reference_2014_merge.shp")]
#
#     x_samples, y_samples = [], []
#     for pair in img_and_shps:
#         name = "/tmp/mzvyagin/"
#         name += "gis_data"
#         name += image_type
#         if large_image:
#             name += "large_image"
#         name += "TFdataset.pkl"
#         if path.exists(name):
#             try:
#                 cache_object = open(name, "rb")
#                 (x_train, y_train), (x_test, y_test) = pickle.load(cache_object)
#                 return (x_train, y_train), (x_test, y_test)
#             except:
#                 print("ERROR: could not load from cache file. Please try removing " + name + " and try again.")
#                 sys.exit()
#         # process each pair and generate the windows
#         else:
#             mask = mask_from_shp(pair[0], pair[1])
#             if image_type == "full_channel":
#                 windows = get_windows(pair[0], mask, large_image)
#             elif image_type == "rgb":
#                 windows = get_rgb_windows(pair[0], mask, large_image)
#             elif image_type == "ir":
#                 windows = get_ir_windows(pair[0], mask, large_image)
#             elif image_type == "hsv":
#                 windows = get_hsv_windows(pair[0], mask, large_image)
#             elif image_type == "hsv_with_ir":
#                 windows = get_hsv_with_ir_windows(pair[0], mask, large_image)
#             elif image_type == "veg_index":
#                 windows = get_vegetation_index_windows(pair[0], mask, large_image)
#             else:
#                 print("WARNING: no image type match, defaulting to RGB+IR")
#                 windows = get_windows(pair[0], mask, large_image)
#             # cache the windows
#             # need to convert to the tensorflow tensors instead of pytorch
#             x, y = [], []
#             for sample in windows:
#                 x.append(pt_to_tf(sample[0]))
#                 y.append(pt_to_tf(sample[1]))
#             x_samples.extend(x)
#             y_samples.extend(y)
#
#     # generate test_train splits
#     x_train, x_test, y_train, y_test = train_test_split(x_samples, y_samples, train_size=0.8, shuffle=False,
#                                                         random_state=0)
#     cache_object = open(name, "wb")
#     #train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
#     #test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
#     pickle.dump(((x_train, y_train), (x_test, y_test)), cache_object)
#     return (x_train, y_train), (x_test, y_test)
