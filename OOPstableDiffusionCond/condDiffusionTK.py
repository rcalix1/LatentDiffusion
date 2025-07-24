## Author: Ricardo A. Calix, Ph.D.
## Stable Diffusion, Conditional latent stable diffusion, towards gameNgen
## Last update July 23, 2025
## Released as is with no warranty
## MIT License

##################################################

import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import tqdm
from tqdm.notebook import trange, tqdm
from torch.optim.lr_scheduler import MultiplicativeLR, LambdaLR
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from einops import rearrange
import math

###################################################

class condDiffusion:

    def __init__(self):
        self.MyName            = 'conditionalDiffusion'
        self.x_means           = None
        self.x_deviations      = None
        self.device            = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size        = 64
        self.sigma             =  25.0
        self.num_steps         = 500
        self.n_epochs          = 100     #{'type':'integer'}
        self.lr                = 10e-4   #{'type':'number'}

        self.transform         = transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5,), (0.5,))
                                  ])
      

    ############################################
    
    def printName(self):
        print( self.MyName  )

    #############################################

    def load_MIST(self):

        self.train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=self.transform, download=True)
        self.train_loader  = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size , shuffle=True)

    #############################################

    def my_view_sample_MNIST(self):
        unique_images, unique_labels = next(iter(self.train_loader))
        unique_images = unique_images.numpy()

        fig, axes = plt.subplots(4, 16, figsize=(16, 4), sharex=True, sharey=True)  # Create a 4x16 grid of subplots with a wider figure

        for i in range(4):  # Loop over rows
            for j in range(16):  # Loop over columns
                index = i * 16 + j  # Calculate the index in the batch
                axes[i, j].imshow(unique_images[index].squeeze(), cmap='gray')  # Show the image using a grayscale colormap
                axes[i, j].axis('off')  # Turn off axis labels and ticks

        plt.show()  # Display the plot

    ##############################################

    # Marginal Probability Standard Deviation Function
    def marginal_prob_std(self, t, sigma):
        """
        Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

        Parameters:
        - t: A vector of time steps.
        - sigma: The $\sigma$ in our SDE.

        Returns:
        - The standard deviation.
        """
        # Convert time steps to a PyTorch tensor
        t = torch.tensor(t, device=self.device)

        # Calculate and return the standard deviation based on the given formula
        return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

    ################################################

    def diffusion_coeff(self, t, sigma):
        """
        Compute the diffusion coefficient of our SDE.

        Parameters:
        - t: A vector of time steps.
        - sigma: The $\sigma$ in our SDE.

        Returns:
        - The vector of diffusion coefficients.
        """
        # Calculate and return the diffusion coefficients based on the given formula
        return torch.tensor(sigma**t, device=self.device)

    #################################################

    def init_marginal_prob_diff_coeff(self):

        # marginal probability standard
        self.marginal_prob_std_fn = functools.partial(self.marginal_prob_std, sigma=self.sigma)

        # diffusion coefficient
        self.diffusion_coeff_fn = functools.partial(self.diffusion_coeff,     sigma=self.sigma)

    ##################################################

    def Euler_Maruyama_sampler(self,
                               score_model,
                               marginal_prob_std,
                               diffusion_coeff,
                               batch_size=64,
                               x_shape=(1, 28, 28),
                               num_steps=500,
                               device=None,
                               eps=1e-3, y=None):
        """
        Generate samples from score-based models with the Euler-Maruyama solver.

        Parameters:
        - score_model: A PyTorch model that represents the time-dependent score-based model.
        - marginal_prob_std: A function that gives the standard deviation of the perturbation kernel.
        - diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
        - batch_size: The number of samplers to generate by calling this function once.
        - x_shape: The shape of the samples.
        - num_steps: The number of sampling steps, equivalent to the number of discretized time steps.
        - device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
        - eps: The smallest time step for numerical stability.
        - y: Target tensor (not used in this function).

        Returns:
        - Samples.
        """

        # Initialize time and the initial sample
        t = torch.ones(batch_size, device=self.device)
        init_x = torch.randn(batch_size, *x_shape, device=self.device) * marginal_prob_std(t)[:, None, None, None]

        # Generate time steps
        time_steps = torch.linspace(1., eps, num_steps, device=self.device)
        step_size = time_steps[0] - time_steps[1]
        x = init_x

        # Sample using Euler-Maruyama method
        with torch.no_grad():
            for time_step in tqdm(time_steps):
                batch_time_step = torch.ones(batch_size, device=self.device) * time_step
                g = diffusion_coeff(batch_time_step)
                mean_x = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step, y=y) * step_size
                x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)

        # Do not include any noise in the last sampling step.
        return mean_x

    #######################################################












        
