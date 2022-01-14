### Ephemeral Stream Segmentation

This code base is used to train computer vision models, namely UNet++, on ultra high resolution aerial imagery in order to perform binary segmentation 
of ephemeral streams. By mapping these streams, we hope to generate information on the environmental impact created by building solar farms. 

This dataset is very large, in the 100s of GB, therefore the model operates on small 256x256 pixel windows of the data. 

This is the latest segmentation we have of the entire dataset, with the windows stitched back together to showcase the power of the trained model. 

![Image of real map and model segmentation](/images/full_segmentation.png?raw=true "Optional Title")
