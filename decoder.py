# This script contains all decoders. 
# Full architectures combining encoders and decoders will be in utils.py.

import torch
import torch.nn as nn

def deconv_block(in_channel, out_channel, kernel=3, stride=2, padding=1, output_padding=1, batch_norm=False):
    base = [
        nn.ConvTranspose2d(in_channel, out_channel, kernel, stride=stride, padding=padding, output_padding=output_padding), # (128, 8, 8)
        nn.LeakyReLU(0.1),
    ]
    if not batch_norm:
        return base
    base = base[:1] + [nn.BatchNorm2d(out_channel)] + base[1:]
    return base

# Simpler blocks so that we don't have to keep calculate kernel, stride, padding. 
# Only specify we want same mapping (H,W) -> (H,W) or double mapping (H,W) -> (2*H,2*W)
def simple_deconv_block(in_c, out_c, double=False, batch_norm=True):
    if double:
        return deconv_block(in_c, out_c, batch_norm=True)
    else:
        return deconv_block(in_c, out_c, kernel=3, stride=1, output_padding=0, batch_norm=True)


class Decoder(nn.Module):
    # Can be used for Vanilla autoencoder or VAE 
    # since variational only affects encoder.
    def __init__(self, latent_dims):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 128),
            nn.ReLU(),
            nn.Linear(128, 3*3*32),
            nn.ReLU(),
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32,3,3))
        self.decoder_conv = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 4, stride=2, padding=1),
        )
    
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x
    
    
class DecoderCIFAR(nn.Module):
    # Can be used for Vanilla autoencoder or VAE 
    # since variational only affects encoder.
    def __init__(self, latent_dims):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 256*4*4*3),
            nn.ReLU(),
        )

        self.raise_1 = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(256,4,4)),
        )
        self.raise_2 = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(64,8,8)),
            nn.Conv2d(64,128,1),
        )
        self.raise_3 = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(16,16,16)),
            nn.Conv2d(16,64,1),
        )
        self.decoder_conv_1 = nn.Sequential( # from (256, 4, 4)
            *simple_deconv_block(256, 256, double=True), # (256, 8, 8)
            *simple_deconv_block(256, 256), # (256, 8, 8)
            *simple_deconv_block(256, 128), # (128, 8, 8)
        )
        self.decoder_conv_2 = nn.Sequential(
            *simple_deconv_block(128, 128, double=True), # (128, 16, 16)
            *simple_deconv_block(128, 64), # (64, 16, 16)
        )
        self.decoder_conv_3 = nn.Sequential(
            *simple_deconv_block(64, 64, double=True), # (64, 32, 32)
            *simple_deconv_block(64, 64), # (64, 32, 32)
            nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1), # (3, 32, 32)
            nn.Sigmoid(),
        ) # Output [N, 3, 32, 32]
    
    
    def forward(self, z):
        z = self.decoder_lin(z)
        # Split latents into global, finer, local details
        z_1, z_2, z_3 = z[:, :4096], z[:, 4096:8192], z[:, 8192:] 

        z_1 = self.raise_1(z_1) # (B, 256, 4, 4)
        z_2 = self.raise_2(z_2) # (B, 128, 8, 8)
        z_3 = self.raise_3(z_3) # (B, 64, 16, 16)

        # Weighted decoding and upscaling
        x = self.decoder_conv_1(z_1)
        x = 0.5 * x + 0.5 * z_2
        x = self.decoder_conv_2(x)
        x = (2/3) * x + (1/3) * z_3
        x = self.decoder_conv_3(x)

        return x
    
class DecoderCelebA(nn.Module):
    # Can be used for Vanilla autoencoder or VAE 
    # since variational only affects encoder.
    def __init__(self, latent_dims):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 256*4*4*4),
            nn.ReLU(),
        )

        self.raise_1 = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(256,4,4)), # (256,4,4)
        )
        self.raise_2 = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(64,8,8)),
            nn.Conv2d(64,256,1), # (256,8,8)
        )
        self.raise_3 = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(16,16,16)),
            nn.Conv2d(16,128,1), # (128,16,16)
        )
        self.raise_4 = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(4,32,32)),
            nn.Conv2d(4,64,1), # (64,32,32)
        )
        self.decoder_conv_1 = nn.Sequential( # from (256, 4, 4)
            *simple_deconv_block(256, 256, double=True), # (256, 8, 8)
            *simple_deconv_block(256, 256), # (256, 8, 8)
        )
        self.decoder_conv_2 = nn.Sequential( # from (256, 8, 8)
            *simple_deconv_block(256, 256, double=True), # (256, 16, 16)
            *simple_deconv_block(256, 256), # (256, 16, 16)
            *simple_deconv_block(256, 128), # (128, 16, 16)
        )
        self.decoder_conv_3 = nn.Sequential( # from (128,16,16)
            *simple_deconv_block(128, 128, double=True), # (128, 32, 32)
            *simple_deconv_block(128, 64), # (64, 32, 32)
        )
        self.decoder_conv_4 = nn.Sequential( # from (64,32,32)
            *simple_deconv_block(64, 64, double=True), # (64, 64, 64)
            *simple_deconv_block(64, 32), # (32,64,64)
            nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1), # (3, 64, 64)
            nn.Sigmoid(),
        ) # Output [N, 3, 64, 64]
    
    
    def forward(self, z):
        z = self.decoder_lin(z)
        # Split latents into global, finer, local details
        z_1, z_2, z_3, z_4 = z[:, :4096], z[:, 4096:8192], z[:, 8192:12288], z[:, 12288:] 

        z_1 = self.raise_1(z_1) # (B, 256, 4, 4)
        z_2 = self.raise_2(z_2) # (B, 256, 8, 8)
        z_3 = self.raise_3(z_3) # (B, 128, 16, 16)
        z_4 = self.raise_4(z_4) # (B, 64, 32, 32)

        # Weighted decoding and upscaling
        x = self.decoder_conv_1(z_1)
        x = 0.5 * x + 0.5 * z_2
        x = self.decoder_conv_2(x)
        x = (2/3) * x + (1/3) * z_3
        x = self.decoder_conv_3(x)
        x = (3/4) * x + (1/4) * z_4
        x = self.decoder_conv_4(x)

        return x
    

class betaDecoder(nn.Module):
    # Can be used for Vanilla autoencoder or VAE 
    # since variational only affects encoder.
    def __init__(self, latent_dims):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 128),
            nn.ReLU(),
            nn.Linear(128, 2*2*64),
            nn.ReLU(),
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(64,2,2))
        self.decoder_conv = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(64, 16, 4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 4, stride=2),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x

class betaDecoderCIFAR(nn.Module):
    # Can be used for Vanilla autoencoder or VAE 
    # since variational only affects encoder.
    def __init__(self, latent_dims):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 256*2*2),
            nn.ReLU(),
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(256,2,2))
        self.decoder_conv = nn.Sequential(
            nn.ReLU(),
            *deconv_block(256, 256, batch_norm=True), # (256, 4, 4)
            *deconv_block(256, 128, batch_norm=True), # (128, 8, 8)
            *deconv_block(128, 64, batch_norm=True), # (64, 16, 16)
            nn.ConvTranspose2d(64, 3, 3, stride=2, padding=1, output_padding=1), # (3, 32, 32)
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x
    
class betaDecoderCelebA(nn.Module):
    # Can be used for Vanilla autoencoder or VAE 
    # since variational only affects encoder.
    def __init__(self, latent_dims):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 256),
            nn.ReLU(),
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(256,1,1))
        self.decoder_conv = nn.Sequential(
            nn.ReLU(),
            *deconv_block(256, 64, kernel=4, stride=1, padding=0, output_padding=0, batch_norm=True), # (128, 8, 8)
            *deconv_block(64, 64, kernel=4, output_padding=0, batch_norm=True), # (64, 16, 16)
            *deconv_block(64, 32, kernel=4, output_padding=0, batch_norm=True),
            *deconv_block(32, 32, kernel=4, output_padding=0, batch_norm=True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), # (3, 32, 32)
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x
    
class MMDDecoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.latent_dims = latent_dims
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 1024),
            nn.ReLU(),
            nn.Linear(1024, 64*2*7*7),
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(128,7,7))
        self.decoder_conv = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x
    
class MMDDecoderCIFAR(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.latent_dims = latent_dims
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256*4*4),
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(256,4,4))
        self.decoder_conv = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x
    

class MMDDecoderCelebA(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.latent_dims = latent_dims
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256*4*4),
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(256,4,4))
        self.decoder_conv = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Sigmoid(),
        )  

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x
    

class optimalSigmaDecoderCIFAR(nn.Module):
    # Can be used for Vanilla autoencoder or VAE 
    # since variational only affects encoder.
    def __init__(self, latent_dims):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 256*4*4),
            nn.ReLU(),
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(256,4,4))
        self.decoder_conv = nn.Sequential(
            nn.ReLU(),
            *deconv_block(256, 128, batch_norm=True), # (128, 8, 8)
            *deconv_block(128, 64, batch_norm=True), # (64, 16, 16)
            nn.ConvTranspose2d(64, 3, 3, stride=2, padding=1, output_padding=1), # (3, 32, 32)
            nn.Sigmoid(),
        )
    
    
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x
    
laggingDecoder = Decoder
laggingDecoderCIFAR = DecoderCIFAR
laggingDecoderCelebA = DecoderCelebA
    
class ResLagDecoderCIFAR(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 16*4*4*3),
            nn.ReLU(),
        )

        self.raise_1 = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(16,4,4)), # (16, 4, 4)
            nn.Conv2d(16,256,1), # (256, 4, 4)
        )
        self.raise_2 = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(4,8,8)),
            nn.Conv2d(4,128,1), # (128, 8, 8)
        )
        self.raise_3 = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(1,16,16)),
            nn.Conv2d(1,64,1), # (64, 16, 16)
        )
        self.decoder_conv_1 = nn.Sequential( # from (256, 4, 4)
            *simple_deconv_block(256, 256, double=True), # (256, 8, 8)
            *simple_deconv_block(256, 256), # (256, 8, 8)
            *simple_deconv_block(256, 128), # (128, 8, 8)
        )
        self.decoder_conv_2 = nn.Sequential(
            *simple_deconv_block(128, 128, double=True), # (128, 16, 16)
            *simple_deconv_block(128, 64), # (64, 16, 16)
        )
        self.decoder_conv_3 = nn.Sequential(
            *simple_deconv_block(64, 64, double=True), # (64, 32, 32)
            *simple_deconv_block(64, 64), # (64, 32, 32)
            nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1), # (3, 32, 32)
            nn.Sigmoid(),
        ) # Output [N, 3, 32, 32]
    
    
    def forward(self, z):
        z = self.decoder_lin(z)
        # Split latents into global, finer, local details
        z_1, z_2, z_3 = z[:, :256], z[:, 256:512], z[:, 512:] 

        z_1 = self.raise_1(z_1) # (B, 256, 4, 4)
        z_2 = self.raise_2(z_2) # (B, 128, 8, 8)
        z_3 = self.raise_3(z_3) # (B, 64, 16, 16)

        # Weighted decoding and upscaling
        x = self.decoder_conv_1(z_1)
        x = 0.5 * x + 0.5 * z_2
        x = self.decoder_conv_2(x)
        x = (2/3) * x + (1/3) * z_3
        x = self.decoder_conv_3(x)

        return x

class ResLagDecoderCelebA(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 64*4*4*4),
            nn.ReLU(),
        )

        self.raise_1 = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(64,4,4)), # (64,4,4)
            nn.Conv2d(64,256,1), # (256,4,4)
        )
        self.raise_2 = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(16,8,8)),
            nn.Conv2d(16,256,1), # (256,8,8)
        )
        self.raise_3 = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(4,16,16)),
            nn.Conv2d(4,128,1), # (128,16,16)
        )
        self.raise_4 = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(1,32,32)),
            nn.Conv2d(1,64,1), # (64,32,32)
        )
        self.decoder_conv_1 = nn.Sequential( # from (256, 4, 4)
            *simple_deconv_block(256, 256, double=True), # (256, 8, 8)
            *simple_deconv_block(256, 256), # (256, 8, 8)
        )
        self.decoder_conv_2 = nn.Sequential( # from (256, 8, 8)
            *simple_deconv_block(256, 256, double=True), # (256, 16, 16)
            *simple_deconv_block(256, 256), # (256, 16, 16)
            *simple_deconv_block(256, 128), # (128, 16, 16)
        )
        self.decoder_conv_3 = nn.Sequential( # from (128,16,16)
            *simple_deconv_block(128, 128, double=True), # (128, 32, 32)
            *simple_deconv_block(128, 64), # (64, 32, 32)
        )
        self.decoder_conv_4 = nn.Sequential( # from (64,32,32)
            *simple_deconv_block(64, 64, double=True), # (64, 64, 64)
            *simple_deconv_block(64, 32), # (32,64,64)
            nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1), # (3, 64, 64)
            nn.Sigmoid(),
        ) # Output [N, 3, 64, 64]
    
    
    def forward(self, z):
        z = self.decoder_lin(z)
        # Split latents into global, finer, local details
        z_1, z_2, z_3, z_4 = z[:, :1024], z[:, 1024:2048], z[:, 2048:3072], z[:, 3072:] 

        z_1 = self.raise_1(z_1) # (B, 256, 4, 4)
        z_2 = self.raise_2(z_2) # (B, 256, 8, 8)
        z_3 = self.raise_3(z_3) # (B, 128, 16, 16)
        z_4 = self.raise_4(z_4) # (B, 64, 32, 32)

        # Weighted decoding and upscaling
        x = self.decoder_conv_1(z_1)
        x = 0.5 * x + 0.5 * z_2
        x = self.decoder_conv_2(x)
        x = (2/3) * x + (1/3) * z_3
        x = self.decoder_conv_3(x)
        x = (3/4) * x + (1/4) * z_4
        x = self.decoder_conv_4(x)

        return x