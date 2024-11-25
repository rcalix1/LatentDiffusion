## Stable Diffusion

* Stable Diffusion for MNIST

## U-Net Models

* Has a U shape from dowsampling to up-sampling the images
* Downsampling: images compressed spatially but expanded channel wise
* Upsampling: images are expanded spatially while the number of channels is reduced
* The U-net has Skip Connections ( this requires a new layer called Concatenate)
* The U-net needs Skip Connections because it is critical that when we Up-sample back to the original image size, that we pass back into each layer the spatial information that was lost during downsampling
* In the U-net we concatenate or connect upsampling layers to the equivalently sized layers in the downsampling part of the network
* These layers are joined together along the channel dimensions so the number of channels increases from k to 2k
* The number of spatial dimensions should remain the same

## Score function

* Learning Score Function
* the approach involves training a neural network to ‘denoise’ samples using the denoising objective

$$
    J = S( x_( noised )  )
$$
