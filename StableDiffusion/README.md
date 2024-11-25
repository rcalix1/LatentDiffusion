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
    J = s( x_{ noised }, t  )  \sigma^2(t) + ( x_{noised} - x_{0})
$$

* Expressing the same idea in a way closer to the actual implementation:

$$
  L2(  J = s( x_{ 0 }  + \sigma(t) \epsilon, t )  \sigma(t) + \epsilon )^2
$$
  
* It’s important to understand the concept that our aim is to predict the amount of noise added to each part of our sample effectively at every time t in the diffusion process
*  and for every x0​ in our original distribution (cars, cats, etc.)
* s(⋅,⋅) represents the score function
* σ(t) is a function of time
* Learning the score function is like transforming random noise into something meaningful
* This loss function figures out how wrong our model is while training.
* It involves picking a random time, getting the noise level, adding this noise to our data,
* and then checking how off our model’s prediction is from reality.
* The aim is to reduce this error during training
* Stable Diffusion creates an image by starting with a totally random one.
* The noise predictor then guesses how noisy the image is,
* and this guessed noise is removed from the image.
* This whole cycle repeats several times, resulting in a clean image at the end
* 
