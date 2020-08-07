### Utility script to convert a .shp file which tags streams into .tif file - black and white mask
import argparse
import geopandas as gpd
import rasterio
from geocube.api.core import make_geocube
import numpy

# command line input

def convert(img_file, shp_file, output_file):
    # this function will perform the conversion
    img = rasterio.open(img_file).read(1)
    streams = gpd.read_file(shp_file)
    streams['exists'] = 1
    # use the image dimensions in order to get resolution for raster
    cube = make_geocube(vector_data=streams, measurements=["exists"],like=img, fill=numpy.nan,).fillna(0)
    cube.rio.to_raster(output_file)


if __name__ == "__main__":
    # set up command line arguments
    parser = argparse.ArgumentParser(description='Convert a shape file into boolean raster file using original image dimensions.')
    parser.add_argument('image_file')
    parser.add_argument('shape_file')
    parser.add_argument('output_file')
    # parse the arguments
    args = parser.parse_args()
    # perform the conversion
    convert(args.image_file, args.shape_file, args.output_file)

