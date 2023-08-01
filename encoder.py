# This script contains all encoders. 
# Full architectures combining encoders and decoders will be in utils.py.

import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import compute_kl_div

# Added kaiming initialization
def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        torch.nn.init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

# Blocks
def conv_block(in_channel, out_channel, kernel_size=3, stride=2, padding=1, batch_norm = False):
    base = [
            nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding), 
            nn.LeakyReLU(0.1),
        ]
    if not batch_norm:
        return base
    base = base[:1] + [nn.BatchNorm2d(out_channel)] + base[1:]
    return base

# Simpler blocks so that we don't have to keep calculate kernel, stride, padding. 
# Only specify we want same mapping (H,W) -> (H,W) or half mapping (H,W) -> (H/2,W/2)
def simple_conv_block(in_c, out_c, half=False, batch_norm=True):
    if half:
        return conv_block(in_c, out_c, kernel_size=4, batch_norm=batch_norm)
    else:
        return conv_block(in_c, out_c, kernel_size=3, stride=1, batch_norm=batch_norm)

# Model architectures
class Encoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.latent_dims = latent_dims
        self.feature_map = nn.Sequential(
            *conv_block(1, 8),
            *conv_block(8, 16, batch_norm=True),
            *conv_block(16, 32, padding=0),
        )
        self.linear_map = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*3*32, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dims) # generate latent
        )
    
    def forward(self, x):
        x = self.feature_map(x)
        z = self.linear_map(x)
        return z

class EncoderCIFAR(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.latent_dims = latent_dims
        # Input is [N, 3, 32, 32]
        self.feature_map = nn.Sequential(
            *conv_block(3, 64, kernel_size=4, batch_norm=True), # (64, 16, 16)
            *conv_block(64, 128, kernel_size=4, batch_norm=True), # (128, 8, 8)
            *conv_block(128, 256, kernel_size=4, batch_norm=True), # (256, 4, 4)
        ) # Output [N, 256, 4, 4]
        self.linear_map = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*4*4, self.latent_dims),
        )
    
    def forward(self, x):
        z = self.feature_map(x)
        z = self.linear_map(z)
        return z

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.feature_map = nn.Sequential(
            *conv_block(1, 8),
            *conv_block(8, 16),
            *conv_block(16, 32, padding=0),
            nn.Flatten(),
            nn.Linear(3*3*32, 128),
            nn.ReLU(),
        )
        self.linear_mu = nn.Linear(128, latent_dims) # Generate mus
        self.linear_sigma = nn.Linear(128, latent_dims) # Generate sigmas

        self.N = torch.distributions.Normal(0,1)
        # Try to use cuda for sampling
        try:
            self.N.loc = self.N.loc.cuda() # Make sampling on GPU
            self.N.scale = self.N.scale.cuda() # Make sampling on GPU
        except: 
            pass
        self.kl = 0 
    
    def forward(self, x):
        x = self.feature_map(x)
        mu = self.linear_mu(x)
        sigma = torch.exp(self.linear_sigma(x))
        z = mu + sigma * self.N.sample(mu.shape)
        # Calculate KL Divergence
        self.kl = compute_kl_div(mu, sigma, reduction="sum")
        return z

class VariationalEncoderCIFAR(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.latent_dims = latent_dims
        self.feature_map_1 = nn.Sequential( # from (3, 32, 32)
            *simple_conv_block(3, 64), # (64, 32, 32)
            *simple_conv_block(64, 64), # (64, 32, 32)
            *simple_conv_block(64, 64, half=True), # (64, 16, 16)
        )    
        self.feature_map_2 = nn.Sequential(
            *simple_conv_block(64, 128), # (128, 16, 16)
            *simple_conv_block(128, 128, half=True), # (128, 8, 8)
        )
        self.feature_map_3 = nn.Sequential(
            *simple_conv_block(128, 256), # (256, 8, 8)
            *simple_conv_block(256, 256), # (256, 8, 8)
            *simple_conv_block(256, 256, half=True), # (256, 4, 4)
            nn.Flatten(),
        ) # Output [N, 256, 4, 4]
        self.compress_1 = nn.Sequential(
            nn.Conv2d(64, 16, 1, bias=True), # (64,16,16) -> (16,16,16)
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.Flatten(),
        )
        self.compress_2 = nn.Sequential(
            nn.Conv2d(128, 64, 1, bias=True), # (128,8,8) -> (64,8,8)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Flatten(),
        )
        self.linear_mu = nn.Linear(256*4*4*3, self.latent_dims)
        self.linear_sigma = nn.Linear(256*4*4*3, self.latent_dims)
        with torch.no_grad():
            # Initialize weights of sigma as zero, avoiding loss exploding.
            self.linear_sigma.weight.fill_(0.)
            self.linear_sigma.bias.fill_(0.)
        self.N = torch.distributions.Normal(0,1)
        # Try to use cuda for sampling
        try:
            self.N.loc = self.N.loc.cuda() # Make sampling on GPU
            self.N.scale = self.N.scale.cuda() # Make sampling on GPU
        except: 
            pass
        self.kl = 0 

    def feature_map(self, x):
        x_1 = self.feature_map_1(x)
        x_2 = self.feature_map_2(x_1)
        x_3_flattened = self.feature_map_3(x_2)
        x_1_flattened = self.compress_1(x_1)
        x_2_flattened = self.compress_2(x_2)
        before_linear = torch.cat([x_1_flattened, x_2_flattened, x_3_flattened], dim=1)
        return before_linear

    def forward(self, x):
        x_before_linear = self.feature_map(x)
        mu = self.linear_mu(x_before_linear)
        # Use clamping to avoid loss exploding
        sigma = torch.exp(torch.clamp(self.linear_sigma(x_before_linear), max=1))
        z = mu + sigma * self.N.sample(mu.shape)
        # Calculate KL Divergence
        self.kl = compute_kl_div(mu, sigma, reduction="sum")
        return z
    
class VariationalEncoderCelebA(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.latent_dims = latent_dims
        self.feature_map_1 = nn.Sequential( # from (3, 64, 64)
            *simple_conv_block(3, 32), # (32, 64, 64)
            *simple_conv_block(32, 64), # (64, 64, 64)
            *simple_conv_block(64, 64, half=True), # (64, 32, 32)
        )    
        self.feature_map_2 = nn.Sequential( 
            *simple_conv_block(64, 128), # (128, 32, 32)
            *simple_conv_block(128, 128, half=True), # (128, 16, 16)
        )
        self.feature_map_3 = nn.Sequential(
            *simple_conv_block(128, 256), # (256, 16, 16)
            *simple_conv_block(256, 256), # (256, 16, 16)
            *simple_conv_block(256, 256, half=True), # (256, 8, 8)
        ) 
        self.feature_map_4 = nn.Sequential(
            *simple_conv_block(256, 256), # (256, 8, 8)
            *simple_conv_block(256, 256, half=True), # (256, 4, 4)
            nn.Flatten(),
        ) # Output [N, 256, 4, 4]
        self.compress_1 = nn.Sequential(
            nn.Conv2d(64, 4, 1, bias=True), # (64,32,32) -> (4,32,32)
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.1),
            nn.Flatten(),
        )
        self.compress_2 = nn.Sequential(
            nn.Conv2d(128, 16, 1, bias=True), # (128,16,16) -> (16,16,16)
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.Flatten(),
        )
        self.compress_3 = nn.Sequential(
            nn.Conv2d(256, 64, 1, bias=True), # (256,8,8) -> (64,8,8)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Flatten(),
        )
        self.linear_mu = nn.Linear(256*4*4*4, self.latent_dims)
        self.linear_sigma = nn.Linear(256*4*4*4, self.latent_dims)
        with torch.no_grad():
            # Initialize weights of sigma as zero, avoiding loss exploding.
            self.linear_sigma.weight.fill_(0.)
            self.linear_sigma.bias.fill_(0.)
        self.N = torch.distributions.Normal(0,1)
        # Try to use cuda for sampling
        try:
            self.N.loc = self.N.loc.cuda() # Make sampling on GPU
            self.N.scale = self.N.scale.cuda() # Make sampling on GPU
        except: 
            pass
        self.kl = 0 

    def feature_map(self, x):
        x_1 = self.feature_map_1(x)
        x_2 = self.feature_map_2(x_1)
        x_3 = self.feature_map_3(x_2)
        x_4_flattened = self.feature_map_4(x_3)
        x_1_flattened = self.compress_1(x_1) 
        x_2_flattened = self.compress_2(x_2)
        x_3_flattened = self.compress_3(x_3)
        before_linear = torch.cat([x_1_flattened, x_2_flattened, x_3_flattened, x_4_flattened], dim=1)
        return before_linear

    def forward(self, x):
        x_before_linear = self.feature_map(x)
        mu = self.linear_mu(x_before_linear)
        # Use clamping to avoid loss exploding
        sigma = torch.exp(torch.clamp(self.linear_sigma(x_before_linear), max=1))
        z = mu + sigma * self.N.sample(mu.shape)
        # Calculate KL Divergence
        self.kl = compute_kl_div(mu, sigma, reduction="sum")
        return z
    
class betaVariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.feature_map = nn.Sequential(
            *conv_block(1, 8),
            *conv_block(8, 16),
            *conv_block(16, 32, padding=0),
            nn.Flatten(),
            nn.Linear(3*3*32, 128),
            nn.ReLU(),
        )
        self.linear_mu = nn.Linear(128, latent_dims) # Generate mus
        self.linear_sigma = nn.Linear(128, latent_dims) # Generate sigmas

        self.N = torch.distributions.Normal(0,1)
        # Try to use cuda for sampling
        try:
            self.N.loc = self.N.loc.cuda() # Make sampling on GPU
            self.N.scale = self.N.scale.cuda() # Make sampling on GPU
        except: 
            pass
        self.kl = 0 

    def forward(self, x):
        x = self.feature_map(x)
        mu = self.linear_mu(x)
        sigma = torch.exp(self.linear_sigma(x))
        z = mu + sigma * self.N.sample(mu.shape)
        # Calculate KL Divergence
        self.kl = compute_kl_div(mu, sigma)
        return z

    
class betaVariationalEncoderCIFAR(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.latent_dims = latent_dims
        self.feature_map = nn.Sequential(
            *conv_block(3, 64, kernel_size=4, batch_norm=True), # (64, 16, 16)
            *conv_block(64, 128, kernel_size=4, batch_norm=True), # (128, 8, 8)
            *conv_block(128, 256, kernel_size=4, batch_norm=True), # (256, 4, 4)
            *conv_block(256, 256, kernel_size=4, batch_norm=True), # (256, 2, 2)
            nn.Flatten(),
        ) # Output [N, 256, 2, 2]
        self.last_linear_mu = nn.Linear(256*2*2, self.latent_dims)
        self.last_linear_sigma = nn.Linear(256*2*2, self.latent_dims)
        with torch.no_grad():
            # Initialize weights of sigma as zero, avoiding loss exploding.
            self.last_linear_sigma.weight.fill_(0.)
            self.last_linear_sigma.bias.fill_(0.)
        self.N = torch.distributions.Normal(0,1)
        # Try to use cuda for sampling
        try:
            self.N.loc = self.N.loc.cuda() # Make sampling on GPU
            self.N.scale = self.N.scale.cuda() # Make sampling on GPU
        except: 
            pass
        self.kl = 0 

    
    def forward(self, x):
        x = self.feature_map(x)
        mu = self.last_linear_mu(x)
        # Use clamping to avoid loss exploding
        sigma = torch.exp(torch.clamp(self.last_linear_sigma(x), max=1))
        z = mu + sigma * self.N.sample(mu.shape)
        # Calculate KL Divergence
        self.kl = compute_kl_div(mu, sigma)
        return z
    
class betaVariationalEncoderCelebA(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.latent_dims = latent_dims
        self.feature_map = nn.Sequential(
            *conv_block(3, 32, kernel_size=4, batch_norm=True), # (32, 32, 32)
            *conv_block(32, 32, kernel_size=4, batch_norm=True), # (32, 16, 16)
            *conv_block(32, 64, kernel_size=4, batch_norm=True), # (64, 8, 8)
            *conv_block(64, 64, kernel_size=4, batch_norm=True), # (64, 4, 4)
            *conv_block(64, 256, kernel_size=4, stride=1, padding=0, batch_norm=True), # (256, 1, 1)
            nn.Flatten(),
        ) # Output [N, 256, 1, 1]
        self.last_linear_mu = nn.Linear(256, self.latent_dims)
        self.last_linear_sigma = nn.Linear(256, self.latent_dims)
        self.N = torch.distributions.Normal(0,1)
        # Try to use cuda for sampling
        try:
            self.N.loc = self.N.loc.cuda() # Make sampling on GPU
            self.N.scale = self.N.scale.cuda() # Make sampling on GPU
        except: 
            pass
        self.kl = 0 
    
    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    
    def forward(self, x):
        x = self.feature_map(x)
        mu = self.last_linear_mu(x)
        # Use clamping to avoid loss exploding
        sigma = torch.exp(torch.clamp(self.last_linear_sigma(x), max=1))
        z = mu + sigma * self.N.sample(mu.shape)
        # Calculate KL Divergence
        self.kl = compute_kl_div(mu, sigma)
        return z
    
class MMDVariationalEncoder(nn.Module):
    # Referencing https://github.com/pratikm141/MMD-Variational-Autoencoder-Pytorch-InfoVAE
    def __init__(self, latent_dims):
        super().__init__()
        self.latent_dims = latent_dims
        self.feature_map = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.1),
            nn.Flatten(),
        )
        self.linear_map = nn.Sequential(
            nn.Linear(64*2*7*7, 1024),
            nn.LeakyReLU(0.1),
        )
        self.linear_mu = nn.Linear(1024, latent_dims)
        self.linear_sigma = nn.Linear(1024, latent_dims)
        self.N = torch.distributions.Normal(0,1)
        # Try to use cuda for sampling
        try:
            self.N.loc = self.N.loc.cuda() # Make sampling on GPU
            self.N.scale = self.N.scale.cuda() # Make sampling on GPU
        except: 
            pass
        self.kl = 0 


    def forward(self, x):
        # Expect input to be (B, 1, 28, 28)
        z = self.feature_map(x)
        z = self.linear_map(z) 
        mu = self.linear_mu(z)
        # Use clamping to avoid loss exploding
        sigma = torch.exp(torch.clamp(self.linear_sigma(z), max=1))
        z = mu + sigma * self.N.sample(mu.shape)
        # Calculate KL Divergence
        self.kl = compute_kl_div(mu, sigma)
        return z
    
class MMDVariationalEncoderCIFAR(nn.Module):
    # Referencing https://github.com/pratikm141/MMD-Variational-Autoencoder-Pytorch-InfoVAE
    def __init__(self, latent_dims):
        super().__init__()
        self.latent_dims = latent_dims
        self.feature_map = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1), # (B, 256, 4, 4)
            nn.Flatten(),
        )
        self.linear_map = nn.Sequential(
            nn.Linear(256*4*4, 1024),
            nn.LeakyReLU(0.1),
        )
        self.linear_mu = nn.Linear(1024, latent_dims)
        self.linear_sigma = nn.Linear(1024, latent_dims)
        self.N = torch.distributions.Normal(0,1)
        # Try to use cuda for sampling
        try:
            self.N.loc = self.N.loc.cuda() # Make sampling on GPU
            self.N.scale = self.N.scale.cuda() # Make sampling on GPU
        except: 
            pass
        self.kl = 0 


    def forward(self, x):
        # Expect input to be (B, 3, 32, 32)
        z = self.linear_map(self.feature_map(x)) 
        mu = self.linear_mu(z)
        # Use clamping to avoid loss exploding
        sigma = torch.exp(torch.clamp(self.linear_sigma(z), max=1))
        z = mu + sigma * self.N.sample(mu.shape)
        # Calculate KL Divergence
        self.kl = compute_kl_div(mu, sigma)
        return z
    
class MMDVariationalEncoderCelebA(nn.Module):
    # Referencing https://github.com/pratikm141/MMD-Variational-Autoencoder-Pytorch-InfoVAE
    def __init__(self, latent_dims):
        super().__init__()
        self.latent_dims = latent_dims
        self.feature_map = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1), 
            nn.Conv2d(256, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1), # (B, 256, 4, 4)
            nn.Flatten(),
        )  
        self.linear_map = nn.Sequential(
            nn.Linear(256*4*4, 1024),
            nn.LeakyReLU(0.1),
        )
        self.linear_mu = nn.Linear(1024, latent_dims)
        self.linear_sigma = nn.Linear(1024, latent_dims)
        self.N = torch.distributions.Normal(0,1)
        # Try to use cuda for sampling
        try:
            self.N.loc = self.N.loc.cuda() # Make sampling on GPU
            self.N.scale = self.N.scale.cuda() # Make sampling on GPU
        except: 
            pass
        self.kl = 0 


    def forward(self, x):
        # Expect input to be (B, 3, 32, 32)
        z = self.linear_map(self.feature_map(x)) 
        mu = self.linear_mu(z)
        # Use clamping to avoid loss exploding
        sigma = torch.exp(torch.clamp(self.linear_sigma(z), max=1))
        z = mu + sigma * self.N.sample(mu.shape)
        # Calculate KL Divergence
        self.kl = compute_kl_div(mu, sigma)
        return z
    

class optimalSigmaVariationalEncoderCIFAR(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.latent_dims = latent_dims
        self.feature_map = nn.Sequential(
            *conv_block(3, 64, kernel_size=4, batch_norm=True), # (64, 16, 16)
            *conv_block(64, 128, kernel_size=4, batch_norm=True), # (128, 8, 8)
            *conv_block(128, 256, kernel_size=4, batch_norm=True), # (256, 4, 4)
            nn.Flatten(),
        ) # Output [N, 256, 4, 4]
        self.last_linear_mu = nn.Linear(256*4*4, self.latent_dims)
        self.last_linear_sigma = nn.Linear(256*4*4, self.latent_dims)
        with torch.no_grad():
            # Initialize weights of sigma as zero, avoiding loss exploding.
            self.last_linear_sigma.weight.fill_(0.)
            self.last_linear_sigma.bias.fill_(0.)
        self.N = torch.distributions.Normal(0,1)
        # Try to use cuda for sampling
        try:
            self.N.loc = self.N.loc.cuda() # Make sampling on GPU
            self.N.scale = self.N.scale.cuda() # Make sampling on GPU
        except: 
            pass
        self.kl = 0 

    
    def forward(self, x):
        x = self.feature_map(x)

        mu = self.last_linear_mu(x)
        # Use clamping to avoid loss exploding
        sigma = torch.exp(torch.clamp(self.last_linear_sigma(x), max=1))
        z = mu + sigma * self.N.sample(mu.shape)
        # Calculate KL Divergence
        self.kl = compute_kl_div(mu, sigma, reduction="sum")
        return z
    


class laggingVariationalEncoder(VariationalEncoder):
    # Lagging VAE, current version
    def __init__(self, latent_dims: int):
        super().__init__(latent_dims)

    def half_forward(self, x):
        # Only return mu and sigma
        x = self.feature_map(x)
        mu = self.linear_mu(x)
        sigma = torch.exp(self.linear_sigma(x))
        return mu, sigma
    
    def forward(self, x):
        mu, sigma = self.half_forward(x)
        z = mu + sigma * self.N.sample(mu.shape)
        # Calculate KL Divergence
        self.kl = compute_kl_div(mu, sigma, reduction="sum")
        return z
    
class laggingVariationalEncoderCIFAR(VariationalEncoderCIFAR):
    # Residual Lagging VAE, old version
    def __init__(self, latent_dims: int):
        super().__init__(latent_dims)

    def half_forward(self, x):
        # Only return mu and sigma
        x = self.feature_map(x)
        mu = self.linear_mu(x)
        sigma = torch.exp(torch.clamp(self.linear_sigma(x), min=-2,max=2))
        return mu, sigma
    
    def forward(self, x):
        mu, sigma = self.half_forward(x)
        z = mu + sigma * self.N.sample(mu.shape)
        # Calculate KL Divergence
        self.kl = compute_kl_div(mu, sigma, reduction="sum")
        return z
    
class laggingVariationalEncoderCelebA(VariationalEncoderCelebA):
    # Residual Lagging VAE, old version
    def __init__(self, latent_dims: int):
        super().__init__(latent_dims)

    def half_forward(self, x):
        # Only return mu and sigma
        x = self.feature_map(x)
        mu = self.linear_mu(x)
        sigma = torch.exp(torch.clamp(self.linear_sigma(x), min=-2,max=2))
        return mu, sigma
    
    def forward(self, x):
        mu, sigma = self.half_forward(x)
        z = mu + sigma * self.N.sample(mu.shape)
        # Calculate KL Divergence
        self.kl = compute_kl_div(mu, sigma, reduction="sum")
        return z

class ResLagVariationalEncoderCIFAR(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.latent_dims = latent_dims
        self.feature_map_1 = nn.Sequential( # from (3, 32, 32)
            *simple_conv_block(3, 64), # (64, 32, 32)
            *simple_conv_block(64, 64), # (64, 32, 32)
            *simple_conv_block(64, 64, half=True), # (64, 16, 16)
        )    
        self.feature_map_2 = nn.Sequential(
            *simple_conv_block(64, 128), # (128, 16, 16)
            *simple_conv_block(128, 128, half=True), # (128, 8, 8)
        )
        self.feature_map_3 = nn.Sequential(
            *simple_conv_block(128, 256), # (256, 8, 8)
            *simple_conv_block(256, 256), # (256, 8, 8)
            *simple_conv_block(256, 256, half=True), # (256, 4, 4)
            *simple_conv_block(256, 16), # (16, 4, 4)
            nn.Flatten(),
        ) # Output [N, 16, 4, 4]
        self.compress_1 = nn.Sequential(
            nn.Conv2d(64, 1, 1, bias=True), # (64,16,16) -> (1,16,16)
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.1),
            nn.Flatten(),
        )
        self.compress_2 = nn.Sequential(
            nn.Conv2d(128, 4, 1, bias=True), # (128,8,8) -> (4,8,8)
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.1),
            nn.Flatten(),
        )
        self.linear_mu = nn.Linear(16*4*4*3, self.latent_dims)
        self.linear_sigma = nn.Linear(16*4*4*3, self.latent_dims)
        with torch.no_grad():
            # Initialize weights of sigma as zero, avoiding loss exploding.
            self.linear_sigma.weight.fill_(0.)
            self.linear_sigma.bias.fill_(0.)
        self.N = torch.distributions.Normal(0,1)
        # Try to use cuda for sampling
        try:
            self.N.loc = self.N.loc.cuda() # Make sampling on GPU
            self.N.scale = self.N.scale.cuda() # Make sampling on GPU
        except: 
            pass
        self.kl = 0 

    def feature_map(self, x):
        x_1 = self.feature_map_1(x)
        x_2 = self.feature_map_2(x_1)
        x_3_flattened = self.feature_map_3(x_2)
        x_1_flattened = self.compress_1(x_1)
        x_2_flattened = self.compress_2(x_2)
        before_linear = torch.cat([x_1_flattened, x_2_flattened, x_3_flattened], dim=1)
        return before_linear

    def forward(self, x):
        x_before_linear = self.feature_map(x)
        mu = self.linear_mu(x_before_linear)
        # Use clamping to avoid loss exploding
        sigma = torch.exp(torch.clamp(self.linear_sigma(x_before_linear), max=1))
        z = mu + sigma * self.N.sample(mu.shape)
        # Calculate KL Divergence
        self.kl = compute_kl_div(mu, sigma, reduction="sum")
        return z

class ResLagVariationalEncoderCelebA(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.latent_dims = latent_dims
        self.feature_map_1 = nn.Sequential( # from (3, 64, 64)
            *simple_conv_block(3, 32), # (32, 64, 64)
            *simple_conv_block(32, 64), # (64, 64, 64)
            *simple_conv_block(64, 64, half=True), # (64, 32, 32)
        )    
        self.feature_map_2 = nn.Sequential( 
            *simple_conv_block(64, 128), # (128, 32, 32)
            *simple_conv_block(128, 128, half=True), # (128, 16, 16)
        )
        self.feature_map_3 = nn.Sequential(
            *simple_conv_block(128, 256), # (256, 16, 16)
            *simple_conv_block(256, 256), # (256, 16, 16)
            *simple_conv_block(256, 256, half=True), # (256, 8, 8)
        ) 
        self.feature_map_4 = nn.Sequential(
            *simple_conv_block(256, 256), # (256, 8, 8)
            *simple_conv_block(256, 256, half=True), # (256, 4, 4)
            *simple_conv_block(256, 64), # (64, 4, 4)
            nn.Flatten(),
        ) # Output [N, 64, 4, 4]
        self.compress_1 = nn.Sequential(
            nn.Conv2d(64, 1, 1, bias=True), # (64,32,32) -> (1,32,32)
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.1),
            nn.Flatten(),
        )
        self.compress_2 = nn.Sequential(
            nn.Conv2d(128, 4, 1, bias=True), # (128,16,16) -> (4,16,16)
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.1),
            nn.Flatten(),
        )
        self.compress_3 = nn.Sequential(
            nn.Conv2d(256, 16, 1, bias=True), # (256,8,8) -> (16,8,8)
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.Flatten(),
        )
        self.linear_mu = nn.Linear(64*4*4*4, self.latent_dims)
        self.linear_sigma = nn.Linear(64*4*4*4, self.latent_dims)
        with torch.no_grad():
            # Initialize weights of sigma as zero, avoiding loss exploding.
            self.linear_sigma.weight.fill_(0.)
            self.linear_sigma.bias.fill_(0.)
        self.N = torch.distributions.Normal(0,1)
        # Try to use cuda for sampling
        try:
            self.N.loc = self.N.loc.cuda() # Make sampling on GPU
            self.N.scale = self.N.scale.cuda() # Make sampling on GPU
        except: 
            pass
        self.kl = 0 

    def feature_map(self, x):
        x_1 = self.feature_map_1(x)
        x_2 = self.feature_map_2(x_1)
        x_3 = self.feature_map_3(x_2)
        x_4_flattened = self.feature_map_4(x_3)
        x_1_flattened = self.compress_1(x_1) 
        x_2_flattened = self.compress_2(x_2)
        x_3_flattened = self.compress_3(x_3)
        before_linear = torch.cat([x_1_flattened, x_2_flattened, x_3_flattened, x_4_flattened], dim=1)
        return before_linear

    def half_forward(self, x):
        # Only return mu and sigma
        x = self.feature_map(x)
        mu = self.linear_mu(x)
        sigma = torch.exp(torch.clamp(self.linear_sigma(x), min=-2,max=2))
        return mu, sigma
    
    def forward(self, x):
        mu, sigma = self.half_forward(x)
        z = mu + sigma * self.N.sample(mu.shape)
        # Calculate KL Divergence
        self.kl = compute_kl_div(mu, sigma, reduction="sum")
        return z