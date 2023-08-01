import torch
import torchvision
from const import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')      
# PREPROCESS = torchvision.models.ResNet152_Weights.IMAGENET1K_V2.transforms()
# RESNET152 = torchvision.models.resnet152(weights="IMAGENET1K_V2").to(DEVICE)
PREPROCESS = torchvision.models.AlexNet_Weights.IMAGENET1K_V1.transforms()
ALEXNET = torchvision.models.alexnet(weights ="IMAGENET1K_V1").to(DEVICE)
ALEXNET = torch.nn.Sequential(*list(ALEXNET.children())[:-1])
ALEXNET.eval()

def reconstruction_loss(prediction, input):
    """The loss is not entirely averaged. It only averaged across the batch size.
    This means, for images with different shape, the maximum possible loss (MPL) will 
    be different. 

    Knowing that image values ranging from 0 to 1, we have the MPL below:
    - MNIST & FashionMNIST: 
            Image size is (1, 28, 28), hence MPL is 28*28=864
    - CIFAR10 & CIFAR100:
            Image size is (3, 32, 32), hence MPL is 3*32*32=3072
    - CelebA:
            Image size after resize is (3, 64, 64), 
                                    hence MPL is 3*64*64 = 12288.
    """
    return ((input - prediction) ** 2).sum() / input.shape[0]

def compute_kl_div(mu, sigma, eps=1e-6, reduction="mean"):
    r"""This loss is entirely averaged, but doesn't have a maximum possible loss (MPL) 
    because the parameters is unbounded (mean and variance of a Gaussian.)
    The loss is entirely averaged in the sense that the shape is (B, z_dim) 
    for both `mu` and `sigma`, and the result is divided by B and z_dim.

    While there is no MPL, the expected value is roughly following the formula below
    .. math::
        \frac12 (\mu^2 + \sigma^2 - 1) - \log(\sigma).
    """
    # Calculate KL Divergence K(q||p)
    # p is the prior (standard gaussian)
    # q is the posterior q(z|x), this function accepts 
    # mu and sigma of q. 
    # Sigma should be standard deviations, not variance, mean it is not squared. 
    # See https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    # for derivations.
    latent_dim = sigma.numel() // sigma.shape[0]
    mean_kl = (0.5*(sigma**2+mu**2-1)-torch.log(torch.clamp(sigma, min=eps))).mean()
    # The above mean_kl is equivalent to KL(q||p) / latent_dim
    if reduction == "mean":
        # Average across every number
        return mean_kl
    elif reduction == "sum":
        # Only average across minibatch dimension.
        return latent_dim * mean_kl

# Below two functions related to MMD VAE, taken from
# https://github.com/pratikm141/MMD-Variational-Autoencoder-Pytorch-InfoVAE
# MMD maximum possible loss is 2 (in the case xx=1, yy=1, xy=0) because it is 
# entirely averaged, but in normal case it won't be 2, will be as small as
# 0.1~0.3, and will be even smaller if batch size and latent dim become bigger. 
# xy won't be exactly 0 because it means kernel compute will give exp(-infty) 
# which is very unlikely. 
def compute_kernel(x, y):
    x_size = x.shape[0]
    y_size = y.shape[0]
    dim = x.shape[1]
    tiled_x = x.view(x_size,1,dim).repeat(1, y_size,1)
    tiled_y = y.view(1,y_size,dim).repeat(x_size, 1,1)
    return torch.exp(-torch.mean((tiled_x - tiled_y)**2,dim=2)/dim*1.0)

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return torch.mean(x_kernel) + torch.mean(y_kernel) - 2*torch.mean(xy_kernel)

def perceptual_loss(prediction, input):
    original_scale = input.shape[1] * input.shape[2] * input.shape[3]
    pred = PREPROCESS(prediction)
    input = PREPROCESS(input)
    pred_feature = ALEXNET(pred)
    input_feature = ALEXNET(input)
    minimum = min(torch.min(pred_feature).item(), torch.min(input_feature).item())
    maximum = max(torch.max(pred_feature).item(), torch.max(input_feature).item())
    pred_feature = (pred_feature - minimum) / (maximum - minimum)
    input_feature = (input_feature - minimum) / (maximum - minimum)
    atomic_loss = torch.mean((pred_feature - input_feature)**2) 
    return original_scale * atomic_loss # Scale to the reconstruction loss.

def mle_variance(t):
    # t is a [B,C,H,W] tensor, this function give variance (sigma^2)
    # where the mean is calculate across entire tensor. 
    mean = t.mean()
    s = t.shape
    d = s[0] * s[1] * s[2] * s[3]
    variance = torch.sum((t-mean) ** 2)/d
    return variance