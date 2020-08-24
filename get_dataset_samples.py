### Utility Script to Generate Sample Images to Visually Test Models

from preprocess import GISDataset
import pickle
from copy import deepcopy

if __name__ == "__main__":
    # files = [("/Users/mzvyagin/Documents/GISProject/nucleus_data/Ephemeral_Channels/Imagery/vhr_2012_refl.img",
    #           "/Users/mzvyagin/Documents/GISProject/nucleus_data/Ephemeral_Channels/Reference/reference_2012_merge.shp"),
    #          ("/Users/mzvyagin/Documents/GISProject/nucleus_data/Ephemeral_Channels/Imagery/vhr_2014_refl.img",
    #           "/Users/mzvyagin/Documents/GISProject/nucleus_data/Ephemeral_Channels/Reference/reference_2014_merge.shp")]
    files = [("/scratch/mzvyagin/Ephemeral_Channels/Imagery/vhr_2012_refl.img",
      "/scratch/mzvyagin/Ephemeral_Channels/Reference/reference_2012_merge.shp"),
     ("/scratch/mzvyagin/Ephemeral_Channels/Imagery/vhr_2014_refl.img",
      "/scratch/mzvyagin/Ephemeral_Channels/Reference/reference_2014_merge.shp")]

    image_types = ['full_channel', 'rgb', 'ir', 'hsv', 'hsv_with_ir', 'veg_index', 'large_full_channel']

    # load up each kind of dataset, and pull a select few images from it
    image_samples = {}
    for i in image_types:
        print("Getting samples from " + i + " dataset...")
        # get specific image type
        if i != 'large_full_channel':
            data = GISDataset(files, i)
        else:
            data = GISDataset(files, 'full_channel', large_image=True)
        samples = [deepcopy(data[1000]), deepcopy(data[2000]), deepcopy(data[3000])]
        # garbage collector helper due to massive size of datasets
        del data
        image_samples[i] = samples

    # serialize the dict and save to pickled file
    f = open("image_samples.pkl", "wb+")
    pickle.dump(image_samples, f)
    print("Image samples [1000, 2000, 3000] have been saved to the file image_samples.pkl.\n")
