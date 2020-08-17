### Utility Script to Generate Sample Images to Visually Test Models

from preprocess import GISDataset
import pickle

if __name__ == "__main__":
    files = [("/Users/mzvyagin/Documents/GISProject/nucleus_data/Ephemeral_Channels/Imagery/vhr_2012_refl.img",
              "/Users/mzvyagin/Documents/GISProject/nucleus_data/Ephemeral_Channels/Reference/reference_2012_merge.shp"),
             ("/Users/mzvyagin/Documents/GISProject/nucleus_data/Ephemeral_Channels/Imagery/vhr_2014_refl.img",
              "/Users/mzvyagin/Documents/GISProject/nucleus_data/Ephemeral_Channels/Reference/reference_2014_merge.shp")]

    image_types = ['full_channel', 'rgb', 'ir', 'hsv', 'hsv_with_ir', 'veg_index']

    # load up each kind of dataset, and pull a select few images from it
    image_samples = {}
    for i in image_types:
        print("Getting samples from " + i + " dataset...")
        # get specific image type
        data = GISDataset(files, i)
        samples = [data[1000], data[2000], data[3000]]
        # garbage collector helper due to massive size of datasets
        del data
        del samples
        image_samples[i] = samples

    # serialize the dict and save to pickled file
    f = open("image_samples.pkl", "w+")
    pickle.dump(image_samples, f)
    print("Image samples [1000, 2000, 3000] have been saved to the file image_samples.pkl.\n")
