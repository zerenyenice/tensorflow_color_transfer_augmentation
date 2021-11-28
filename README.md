# tensorflow_color_transfer_augmentation
Masked reinhard color transfer augmentation implemented for using on tensorflow dataset pipeline.

This tensorflow implementation is mostly based on to the "Color Transfer between Images" paper by Reinhard et al. (2001).

When you have two images loaded with BGR color space, it transfers the color distribution from the target image to the source
image by using the mean and standard deviations of the (Lab) color space.


