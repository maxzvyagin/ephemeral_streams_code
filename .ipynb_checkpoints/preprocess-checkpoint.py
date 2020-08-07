### Module in order to preprocess the satellite images and ready them for input into PyTorch
### Max Zvyagin

import rasterio
import rasterio.features
import geopandas as gpd
import torch
from torch.utils.data import Dataset


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


# given the name of an image file and the corresponding .shp array mask, outputs an array of image windows and mask windows
def get_windows(img_f, mask):
    samples = []
    with rasterio.open(img_f) as src:
        for ji, window in src.block_windows():
            # get the window from the mask
            mask_check = mask[window.row_off:window.row_off + window.height,
                         window.col_off:window.col_off + window.width]
            if True in mask_check:
                r = src.read(window=window)
                # also can probably convert to tensors here as well
                samples.append((torch.from_numpy(r), torch.from_numpy(mask_check)))
            else:
                pass
    return samples


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
