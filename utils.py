import torchvision
from torchvision import transforms
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from numpy.random import default_rng
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import json
import numpy as np
from tqdm import tqdm
import importlib
import encoder, decoder
from encoder import *
from decoder import *
from loss import *
from const import *
# torch.autograd.set_detect_anomaly(True)

# This program includes software developed by jterrace and David Kim
# in https://stackoverflow.com/questions/10097477/python-json-array-newlines
# Huge thanks to them!
# Changed basestring to str, and dict uses items() instead of iteritems().
def to_json(o, level=0):
    # Common utilities
    INDENT = 4
    SPACE = " "
    NEWLINE = "\n"
    ret = ""
    if isinstance(o, dict):
        ret += NEWLINE + SPACE * INDENT * level + "{" + NEWLINE
        comma = ""
        for k, v in o.items():
            ret += comma
            comma = ",\n"
            ret += SPACE * INDENT * (level + 1)
            ret += '"' + str(k) + '":' + SPACE
            ret += to_json(v, level + 1)

        ret += NEWLINE + SPACE * INDENT * level + "}"
    elif isinstance(o, str):
        if o.strip().startswith('"') and o.strip().endswith('"'):
            ret += o.strip()
        else:
            ret += '"' + o.strip() + '"'
    elif isinstance(o, list):
        ret += "[" + ",".join([to_json(e, level + 1) for e in o]) + "]"
    # Tuples are interpreted as lists
    elif isinstance(o, tuple):
        ret += "[" + ",".join(to_json(e, level + 1) for e in o) + "]"
    elif isinstance(o, bool):
        ret += "true" if o else "false"
    elif isinstance(o, int):
        ret += str(o)
    elif isinstance(o, float):
        ret += '%.7g' % o
    elif o is None:
        ret += 'null'
    else:
        raise TypeError("Unknown type '%s' for json serialization" %
                        str(type(o)))
    return ret

def pretty_dump(data: dict, filename: str):
    # Make sure parent directory exist.
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    json_string = to_json(data)
    with open(filename, "w") as f:
        f.write(json_string)

def validate_dict(text: str) -> bool: 
    try:
        json_dict = json.loads(text)
    except Exception as e:
        print("JSON loading fault: ", e)
        return False
    

def dataset_loader_atomic(dataset, train_or_valid, normalize=False, n=None):
    # normalize can be True or False. 
    # Set to false when only want to use it to visualize
    # Set to true when want to use for training.
    if normalize: print(f"====Dataset {dataset} will be normalized====")
    if dataset in ["MNIST", "FashionMNIST", "CIFAR10", "CIFAR100"]:
        transform = NORMALIZING_TRANSFORMS[dataset] if normalize\
                    else transforms.ToTensor()
        datas = getattr(torchvision.datasets, dataset)(
            root = "./",
            train = train_or_valid == "train", 
            download = True,
            transform = transform,
        )

    elif dataset in ["CelebA"]:
        transform = NORMALIZING_TRANSFORMS[dataset] if normalize\
                    else transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize((64, 64)),
                    ])
        datas = getattr(torchvision.datasets, dataset)(
            root = CELEBA_DATAROOT,
            split = train_or_valid, 
            download = CELEBA_DOWNLOAD,
            transform = transform
        )
    if TEST_RUN:
        perm = torch.randperm(len(datas))
        datas = [datas[p] for p in perm[:TEST_RUN_TAKE]]
    if n is not None:
        datas = [datas[i] for i in range(n)]
    return datas
    

def dataset_loader(dataset, batch_size: int=16, verbose=False, normalize=TRAINING_NORMALIZE):
    # We assume this function will only be used for training/validation, 
    # For the purpose of visualization, we recommend use atomic loader only.
    train_dataset = dataset_loader_atomic(dataset, "train", normalize=normalize)
    valid_dataset = dataset_loader_atomic(dataset, "valid", normalize=normalize)
    if verbose:
        print(f"{dataset}: Sample training image: {train_dataset[100][0].shape}")
        print(f"{dataset}: Sample validating image: {valid_dataset[100][0].shape}")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    return train_dataset, valid_dataset, train_dataloader, valid_dataloader

def revive_legacy(model_name: str):
    """Reuse older version of encoder and decoder from the script 
    encoder.py and decoder.py stored in the trained model directory
    instead of using the current encoder and decoder scripts.

    Args:
        model_name (str): Model name

    Raises:
        ValueError: _description_
        NotImplementedError: _description_
        NotImplementedError: _description_

    Returns:
        Encoder, Decoder classes.
    """
    model_dir = os.path.join(OUTPUT_ROOT, model_name)
    config_file = os.path.join(model_dir, "config.json")
    with open(config_file, "r") as f:
        data = json.load(f)
    model_code = data["model_code"]
    dataset = data["dataset"]
    encoder_classname = CLASS_MAPPING[(model_code, dataset)][0]
    decoder_classname = CLASS_MAPPING[(model_code, dataset)][1]
    E = importlib.import_module(f"models.{model_name}.encoder")
    D = importlib.import_module(f"models.{model_name}.decoder")
    print("===Using Legacy Encoder and Decoder===")
    return getattr(E, encoder_classname), getattr(D, decoder_classname)

def tensor_to_collage(batched_tensor, nrow=10):
    # Receive a tensor of shape [50, 1, H, W] or [50, 3, H, W]
    # Return a collage of [1, 5H, 10W] or [3, 5H, 10W]    
    collage = torchvision.utils.make_grid(
        batched_tensor, nrow=nrow, padding=2
    )
    collage = collage.cpu().permute(1, 2, 0)
    collage = collage.detach().numpy()

    return collage

def parse_additional_hp(args: dict):
    # Code below is too dirty but as long as it works, heh.
    # Unpack args["additional_hp"] into sensible values
    if args["additional_hp"].strip():
        additional_hp = json.loads(args["additional_hp"])
    else:
        additional_hp = {}
    if type(additional_hp) == str and additional_hp.strip():
        additional_hp = json.loads(additional_hp)
    else:
        additional_hp = {}
    print("\tAdditional Hyperparameters:")
    for key, value in additional_hp.items():
        if key not in args: # Avoid injection that overwrites other important values
            args[key] = value
            print(key, ":", value)
    return args

def kaiming_init(module):
    if not KAIMING_INIT:
        return 
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

def unnormalize(prediction, dataset):
    # If TRAINING_NORMALIZE is true, unnormalize the prediction.
    if TRAINING_NORMALIZE:
        mean, std = MEAN_STD[dataset]
        new_mean = [-m/s for m, s in zip(mean, std)]
        new_std = [1/s for s in std]
        return transforms.Normalize(mean=new_mean, std=new_std)(prediction)
    return prediction

def model_mb_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

# By matthew_zeng in https://discuss.pytorch.org/t/two-optimizers-for-one-model/11085/6
class MultipleOptimizer(object):
    def __init__(self, enc_optimizer, dec_optimizer):
        self.optimizers = [enc_optimizer, dec_optimizer]

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()
    
    def step_enc(self):
        self.optimizers[0].step()

    def step_dec(self):
        self.optimizers[1].step()

    def state_dict(self):
        return {
            "encoder": self.optimizers[0].state_dict(),
            "decoder": self.optimizers[1].state_dict(),
        }
    
    def load_state_dict(self, state_dict):
        self.optimizers[0].load_state_dict(state_dict["encoder"])
        self.optimizers[1].load_state_dict(state_dict["decoder"])


class AutoEncoder(nn.Module):
    def __init__(self, latent_dim, model_name = None):
        super().__init__()
        if model_name is not None:
            Encoder, Decoder = revive_legacy(model_name)
            self.encoder = Encoder(latent_dim)
            self.decoder = Decoder(latent_dim)
        else:
            self.encoder = encoder.Encoder(latent_dim)
            self.decoder = decoder.Decoder(latent_dim)
        kaiming_init(self)

    def forward(self, x):
        z = self.encoder(x)
        output = self.decoder(z)
        return output
    
class AutoEncoderCIFAR(nn.Module):
    def __init__(self, latent_dim, model_name=None):
        super().__init__()
        if model_name is not None:
            EncoderCIFAR, DecoderCIFAR = revive_legacy(model_name)
            self.encoder = EncoderCIFAR(latent_dim)
            self.decoder = DecoderCIFAR(latent_dim)
        else:
            self.encoder = encoder.EncoderCIFAR(latent_dim)
            self.decoder = decoder.DecoderCIFAR(latent_dim)
        kaiming_init(self)


    def forward(self, x):
        z = self.encoder(x)
        output = self.decoder(z)
        return output

class VAE(nn.Module):
    def __init__(self, latent_dim, model_name=None):
        super().__init__()
        if model_name is not None:
            VariationalEncoder, Decoder = revive_legacy(model_name)
            self.vencoder = VariationalEncoder(latent_dim)
            self.decoder = Decoder(latent_dim)
        else:
            self.vencoder = encoder.VariationalEncoder(latent_dim)
            self.decoder = decoder.Decoder(latent_dim)
        kaiming_init(self)

    def forward(self, x):
        z = self.vencoder(x)
        output = self.decoder(z)
        return output
    
class VAECIFAR(nn.Module):
    def __init__(self, latent_dim, model_name = None):
        super().__init__()
        if model_name is not None:
            VariationalEncoderCIFAR, DecoderCIFAR = revive_legacy(model_name)
            self.vencoder = VariationalEncoderCIFAR(latent_dim)
            self.decoder = DecoderCIFAR(latent_dim)
        else:
            self.vencoder = encoder.VariationalEncoderCIFAR(latent_dim)
            self.decoder = decoder.DecoderCIFAR(latent_dim)
        kaiming_init(self)

    def forward(self, x):
        z = self.vencoder(x)
        output = self.decoder(z)
        return output
    
class VAECelebA(nn.Module):
    def __init__(self, latent_dim, model_name = None):
        super().__init__()
        if model_name is not None:
            VariationalEncoderCelebA, DecoderCelebA = revive_legacy(model_name)
            self.vencoder = VariationalEncoderCelebA(latent_dim)
            self.decoder = DecoderCelebA(latent_dim)
        else:
            self.vencoder = encoder.VariationalEncoderCelebA(latent_dim)
            self.decoder = decoder.DecoderCelebA(latent_dim)
        kaiming_init(self)

    def forward(self, x):
        z = self.vencoder(x)
        output = self.decoder(z)
        return output
    
class betaVAE(nn.Module):
    def __init__(self, latent_dim, beta, capacity=None, model_name=None):
        # If capacity is not None, use betaVAe with control capacity instead. 
        # capacity will increase from 0 linearly until final epoch is reached. 
        super().__init__()
        # When beta=1, betaVAE becomes VAE.
        self.beta = beta
        self.capacity = capacity
        if model_name is not None:
            betaVariationalEncoder, betaDecoder = revive_legacy(model_name)
            self.vencoder = betaVariationalEncoder(latent_dim)
            self.decoder = betaDecoder(latent_dim)
        else:
            self.vencoder = encoder.betaVariationalEncoder(latent_dim)
            self.decoder = decoder.betaDecoder(latent_dim)
        kaiming_init(self)
    
    def forward(self, x):
        z = self.vencoder(x)
        output = self.decoder(z)
        return output
    
class betaVAECIFAR(nn.Module):
    def __init__(self, latent_dim, beta, capacity=None, model_name=None):
        # If capacity is not None, use betaVAe with control capacity instead. 
        # capacity will increase from 0 linearly until final epoch is reached. 
        super().__init__()
        # When beta=1, betaVAE becomes VAE.
        self.beta = beta
        self.capacity = capacity
        if model_name is not None:
            betaVariationalEncoderCIFAR, betaDecoderCIFAR = revive_legacy(model_name)
            self.vencoder = betaVariationalEncoderCIFAR(latent_dim)
            self.decoder = betaDecoderCIFAR(latent_dim)
        else:
            self.vencoder = encoder.betaVariationalEncoderCIFAR(latent_dim)
            self.decoder = decoder.betaDecoderCIFAR(latent_dim)
        kaiming_init(self)
    
    def forward(self, x):
        z = self.vencoder(x)
        output = self.decoder(z)
        return output
    
class betaVAECelebA(nn.Module):
    def __init__(self, latent_dim, beta, capacity=None, model_name=None):
        # If capacity is not None, use betaVAe with control capacity instead. 
        # capacity will increase from 0 linearly until final epoch is reached. 
        super().__init__()
        # When beta=1, betaVAE becomes VAE.
        self.beta = beta
        self.capacity = capacity
        if model_name is not None:
            betaVariationalEncoderCelebA, betaDecoderCelebA = revive_legacy(model_name)
            self.vencoder = betaVariationalEncoderCelebA(latent_dim)
            self.decoder = betaDecoderCelebA(latent_dim)
        else:
            self.vencoder = encoder.betaVariationalEncoderCelebA(latent_dim)
            self.decoder = decoder.betaDecoderCelebA(latent_dim)
        kaiming_init(self)
    
    def forward(self, x):
        z = self.vencoder(x)
        output = self.decoder(z)
        return output
    
class MMDVAE(nn.Module):
    def __init__(self, latent_dim, alpha, l, model_name=None):
        super().__init__()
        # When alpha=0, l=1, MMDVAE becomes VAE.
        # When alpha in (0,1) and alpha+l-1=0, MMDVAE becomes beta-VAE with beta=1-alpha.
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.l = l
        if model_name is not None:
            MMDVariationalEncoder, MMDDecoder = revive_legacy(model_name)
            self.vencoder = MMDVariationalEncoder(latent_dim)
            self.decoder = MMDDecoder(latent_dim)
        else:
            self.vencoder = encoder.MMDVariationalEncoder(latent_dim)
            self.decoder = decoder.MMDDecoder(latent_dim)
        kaiming_init(self)
    def forward(self, x):
        z = self.vencoder(x)
        output = self.decoder(z)
        return output
    
class MMDVAECIFAR(nn.Module):
    def __init__(self, latent_dim, alpha, l, model_name=None):
        super().__init__()
        # When alpha=0, l=1, MMDVAE becomes VAE.
        # When alpha in (0,1) and alpha+l-1=0, MMDVAE becomes beta-VAE with beta=1-alpha.
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.l = l
        if model_name is not None:
            MMDVariationalEncoderCIFAR, MMDDecoderCIFAR = revive_legacy(model_name)
            self.vencoder = MMDVariationalEncoderCIFAR(latent_dim)
            self.decoder = MMDDecoderCIFAR(latent_dim)
        else:
            self.vencoder = encoder.MMDVariationalEncoderCIFAR(latent_dim)
            self.decoder = decoder.MMDDecoderCIFAR(latent_dim)
        kaiming_init(self)
    def forward(self, x):
        z = self.vencoder(x)
        output = self.decoder(z)
        return output


class MMDVAECelebA(nn.Module):
    def __init__(self, latent_dim, alpha, l, model_name=None):
        super().__init__()
        # When alpha=0, l=1, MMDVAE becomes VAE.
        # When alpha in (0,1) and alpha+l-1=0, MMDVAE becomes beta-VAE with beta=1-alpha.
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.l = l
        if model_name is not None:
            MMDVariationalEncoderCelebA, MMDDecoderCelebA = revive_legacy(model_name)
            self.vencoder = MMDVariationalEncoderCelebA(latent_dim)
            self.decoder = MMDDecoderCelebA(latent_dim)
        else:
            self.vencoder = encoder.MMDVariationalEncoderCelebA(latent_dim)
            self.decoder = decoder.MMDDecoderCelebA(latent_dim)
        kaiming_init(self)
    def forward(self, x):
        z = self.vencoder(x)
        output = self.decoder(z)
        return output

optimalSigmaVAE = VAE
optimalSigmaVAECIFAR = VAECIFAR
optimalSigmaVAECelebA = VAECelebA

class OptimalSigmaLadderVAEMNIST(nn.Module):
    def __init__(self, latent_dim , model_name = None):
        super().__init__()
        if model_name is not None:
            LadderVariationalEncoder, LadderDecoderMNIST = revive_legacy(model_name)
            self.decoder = LadderDecoderMNIST(latent_dim)
            self.vencoder = LadderVariationalEncoder(self.decoder)
        else:
            self.decoder = decoder.LadderDecoderMNIST(latent_dim)
            self.vencoder = encoder.LadderVariationalEncoder(self.decoder)
        kaiming_init(self)
    
    def forward(self, x):
        z = self.vencoder(x, self.decoder)
        x_hat = self.decoder(z)
        return x_hat
    
class OptimalSigmaLadderVAECIFAR(nn.Module):
    def __init__(self, latent_dim , model_name = None):
        super().__init__()
        if model_name is not None:
            LadderVariationalEncoder, LadderDecoderCIFAR = revive_legacy(model_name)
            self.decoder = LadderDecoderCIFAR(latent_dim)
            self.vencoder = LadderVariationalEncoder(self.decoder)
        else:
            self.decoder = decoder.LadderDecoderCIFAR(latent_dim)
            self.vencoder = encoder.LadderVariationalEncoder(self.decoder)
        kaiming_init(self)
    
    def forward(self, x):
        z = self.vencoder(x, self.decoder)
        x_hat = self.decoder(z)
        return x_hat
    
class laggingVAEbase(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = self.vencoder(x)
        x_hat = self.decoder(z)
        return x_hat
    
    def infer(self, x):
        return self.vencoder(x)
    
    def generate(self, z):
        return self.decoder(z)
    
    def get_posterior(self, x):
        return self.vencoder.half_forward(x)
    
    def zero_kl(self):
        # Clear accumulated kl
        self.kl_div = 0
    
    def marginal_posterior_estimator(self, dataloader, dataset, sample: int=500) -> torch.Tensor:
        """This estimate the term 
        D_{KL} (q_\phi (z) || p(z))
        from the Lagging inference network paper. 

        Args:
            dataloader: Usually a PyTorch Dataloader for the validation set.
            dataset: A pytorch dataset used for sampling x. 
            sample: Number of sample used for evaluate the term.

        Returns:
            torch.Tensor: The KL Divergence stated above.
        """
        rng = default_rng()
        # Force to be eval mode
        self.eval()
        # In case dataloader doesn't even have 500 data
        bs = dataloader.batch_size
        n = len(dataloader.sampler)
        sample = min(sample, len(dataloader.sampler))
        # Sample a minibatch of S latents by ancestral sampling
        latents = []
        indices = rng.choice(list(range(n)), size=sample, replace=False)
        for batch_ind in range(0, len(indices), bs):
            this_inds = indices[batch_ind: batch_ind+bs]
            image_batch = torch.stack([dataset[i][0] for i in this_inds]).to(DEVICE)
            latent_batch = self.infer(image_batch)
            latents.append(latent_batch)
        latents = torch.cat(latents)
        latents = latents.unsqueeze(0)

        # Calculate q_\phi(z_s), scalable computation
        marginal_log_ps = torch.zeros((sample)).to(DEVICE)
        prior_log_ps = Normal(torch.zeros_like(latents), torch.ones_like(latents)).log_prob(latents)
        prior_log_ps = torch.sum(prior_log_ps, dim=(0,2))
        num_batches = len(dataloader)
        it = iter(dataloader)
        desc = "Evaluating marginal posterior: "
        for _ in tqdm(range(0, num_batches), 
                desc=desc):
            # Extract input from dataloader
            input = next(it)[0]
            input = input.to(DEVICE)
            mu, sigma = self.vencoder.half_forward(input)
            mu, sigma = mu.unsqueeze(1), sigma.unsqueeze(1)
            # Clamp sigma
            sigma = torch.clamp(sigma, min=1e-6)
            dist = Normal(mu, sigma)
            # Compute the log probability density of z under the normal distribution
            log_prob = dist.log_prob(latents) / n # q(x_i) = 1/n
            log_prob = torch.sum(log_prob, dim=(0,2))
            # Increment into mps
            marginal_log_ps += log_prob
        # print("Prior log prob shape", prior_log_ps.shape)
        # print("\tmean is", prior_log_ps.mean())
        # print("Marginal Posterior log prob shape", marginal_log_ps.shape)
        # print("\tmean is", marginal_log_ps.mean())
        marginal_log_ps = marginal_log_ps - prior_log_ps
        return torch.mean(marginal_log_ps)
    
    def mutual_info(self, dataloader, dataset, kl_div, sample: int=500):
        # Follow equation 5 in 
        # https://arxiv.org/pdf/1901.05534.pdf
        mpe = self.marginal_posterior_estimator(dataloader, dataset, sample=sample)
        # print("Marginal Posterior Estimate is:", mpe.item())
        return kl_div - mpe
    
class laggingVAE(laggingVAEbase):
    def __init__(self, latent_dim, model_name = None):
        super().__init__()
        if model_name is not None:
            laggingVariationalEncoder, laggingDecoder = revive_legacy(model_name)
            self.vencoder = laggingVariationalEncoder(latent_dim) 
            self.decoder = laggingDecoder(latent_dim)
        else:
            self.vencoder = encoder.laggingVariationalEncoder(latent_dim)
            self.decoder = decoder.laggingDecoder(latent_dim)
        kaiming_init(self)
        
class laggingVAECIFAR(laggingVAEbase):
    def __init__(self, latent_dim, model_name = None):
        super().__init__()
        if model_name is not None:
            laggingVariationalEncoderCIFAR, laggingDecoderCIFAR = revive_legacy(model_name)
            self.vencoder = laggingVariationalEncoderCIFAR(latent_dim) 
            self.decoder = laggingDecoderCIFAR(latent_dim)
        else:
            self.vencoder = encoder.laggingVariationalEncoderCIFAR(latent_dim)
            self.decoder = decoder.laggingDecoderCIFAR(latent_dim)
        kaiming_init(self)

class laggingVAECelebA(laggingVAEbase):
    def __init__(self, latent_dim, model_name = None):
        super().__init__()
        if model_name is not None:
            laggingVariationalEncoderCelebA, laggingDecoderCelebA = revive_legacy(model_name)
            self.vencoder = laggingVariationalEncoderCelebA(latent_dim) 
            self.decoder = laggingDecoderCelebA(latent_dim)
        else:
            self.vencoder = encoder.laggingVariationalEncoderCelebA(latent_dim)
            self.decoder = decoder.laggingDecoderCelebA(latent_dim)
        kaiming_init(self)
    
class ResLagVAECIFAR(laggingVAEbase):
    def __init__(self, latent_dim, model_name = None):
        super().__init__()
        if model_name is not None:
            ResLagVariationalEncoderCIFAR, ResLagDecoderCIFAR = revive_legacy(model_name)
            self.vencoder = ResLagVariationalEncoderCIFAR(latent_dim) 
            self.decoder = ResLagDecoderCIFAR(latent_dim)
        else:
            self.vencoder = encoder.ResLagVariationalEncoderCIFAR(latent_dim)
            self.decoder = decoder.ResLagDecoderCIFAR(latent_dim)
        kaiming_init(self)

class ResLagVAECelebA(laggingVAEbase):
    def __init__(self, latent_dim, model_name = None):
        super().__init__()
        if model_name is not None:
            ResLagVariationalEncoderCelebA, ResLagDecoderCelebA = revive_legacy(model_name)
            self.vencoder = ResLagVariationalEncoderCelebA(latent_dim) 
            self.decoder = ResLagDecoderCelebA(latent_dim)
        else:
            self.vencoder = encoder.ResLagVariationalEncoderCelebA(latent_dim)
            self.decoder = decoder.ResLagDecoderCelebA(latent_dim)
        kaiming_init(self)

def init_model(latent_dim: int, dataset: str, architecture: str, device: str, 
                legacy_model_name: str) -> nn.Module:
    """Initialize model based on dataset configurations.

    Args:
        latent_dim (int): Latent dimension.
        dataset (str): String to indicate dataset being used. 
                    Acceptable: "MNIST", "FashionMNIST", "CIFAR10", "CIFAR100", "CelebA" 
        architecture (str): Model architecture.
                    Acceptable: "ae", "vae", "beta-vae", and more
        device (str): cpu or gpu.
        legacy_model_name (str): Use the code stored in the model.

    Returns:
        nn.Module: Return initialized model 
    """
    config_file = os.path.join(OUTPUT_ROOT, legacy_model_name, "config.json")
    with open(config_file, "r") as f:
        args = json.load(f)
    # args = parse_additional_hp(args)
    model = None
    if dataset in ["MNIST", "FashionMNIST"]:
        if architecture == "ae":
            model = AutoEncoder(latent_dim, model_name=legacy_model_name).to(device)
        elif architecture == "vae":
            model = VAE(latent_dim, model_name=legacy_model_name).to(device)
        elif architecture == "beta-vae":
            model = betaVAE(latent_dim, beta=args["beta"], 
                            capacity = None if "capacity" not in args else args["capacity"], 
                            model_name=legacy_model_name).to(device)
        elif architecture == "mmd-vae":
            model = MMDVAE(latent_dim, alpha=args["alpha"], 
                        l=args["lambda"],model_name=legacy_model_name).to(device)
        elif architecture == "osigma-vae":
            # Optimal Sigma VAE use the same architecture as vanilla VAE,
            # they just interpret ELBO objective differently.
            model = optimalSigmaVAE(latent_dim, model_name=legacy_model_name).to(device)
        elif architecture == "osigma-ladder":
            # Optimal Sigma Ladder VAE
            model = OptimalSigmaLadderVAEMNIST(latent_dim, model_name=legacy_model_name).to(device)
        elif architecture == "lagging-vae":
            model = laggingVAE(latent_dim, model_name = legacy_model_name).to(device)
    elif dataset in ["CIFAR10", "CIFAR100"]:
        if architecture == "ae":
            model = AutoEncoderCIFAR(latent_dim, model_name=legacy_model_name).to(device)
        elif architecture == "vae":
            model = VAECIFAR(latent_dim, model_name=legacy_model_name).to(device)
        elif architecture == "beta-vae":
            model = betaVAECIFAR(latent_dim, beta=args["beta"],
                            capacity = None if "capacity" not in args else args["capacity"],
                            model_name=legacy_model_name).to(device)
        elif architecture == "mmd-vae":
            model = MMDVAECIFAR(latent_dim, alpha=args["alpha"], 
                        l=args["lambda"],model_name=legacy_model_name).to(device)
        elif architecture == "osigma-vae":
            # Optimal Sigma VAE use the same architecture as vanilla VAE,
            # they just interpret ELBO objective differently.
            model = optimalSigmaVAECIFAR(latent_dim, model_name=legacy_model_name).to(device)
        elif architecture == "osigma-ladder":
            # Optimal Sigma Ladder VAE
            model = OptimalSigmaLadderVAECIFAR(latent_dim, model_name=legacy_model_name).to(device)
        elif architecture == "lagging-vae":
            model = laggingVAECIFAR(latent_dim, model_name = legacy_model_name).to(device)
        elif architecture == "reslag-vae":
            model = ResLagVAECIFAR(latent_dim, model_name = legacy_model_name).to(device)
    elif dataset in ["CelebA"]:
        if architecture == "beta-vae":
            model = betaVAECelebA(latent_dim, beta=args["beta"], 
                            capacity = None if "capacity" not in args else args["capacity"],
                            model_name=legacy_model_name).to(device)
        if architecture == "mmd-vae":
            model = MMDVAECelebA(latent_dim, alpha=args["alpha"], 
                        l=args["lambda"],model_name=legacy_model_name).to(device)
        elif architecture == "osigma-vae":
            # Optimal Sigma VAE use the same architecture as vanilla VAE,
            # they just interpret ELBO objective differently.
            model = optimalSigmaVAECelebA(latent_dim, model_name=legacy_model_name).to(device)
        elif architecture == "lagging-vae":
            model = laggingVAECelebA(latent_dim, model_name = legacy_model_name).to(device)
        elif architecture == "reslag-vae":
            model = ResLagVAECelebA(latent_dim, model_name = legacy_model_name).to(device)
    else:
        raise ValueError(f"Couldn't load model because dataset {dataset} is not implemented yet.")
    if model is None:
        raise NotImplementedError(f"Model is not implemented yet. " +\
                                    f"\n\tModelInfo: [{architecture}]" +\
                                    f"\n\tDatasetInfo: [{dataset}]")
    return model


def dump_tensor_to_numpy(tensor_image):
    # the tensor_image should be (C, H, W) 
    # If C=1, output will be (H, W) else will be (H, W, C).
    numpy_image = tensor_image.cpu().detach().numpy()
    if numpy_image.shape[0] == 1:
        numpy_image = numpy_image[0]
    else:
        numpy_image = np.transpose(numpy_image, axes=[1,2,0])
    return numpy_image





def train(model, model_code: str, dataloader, optimizer, device: str, epoch: int = None, total_epoch: int = None,
            others: dict = None) -> dict:
    """General train function

    Args:
        model (_type_): PyTorch model, can be randomly initialized or resume. 
        model_code (str): Model architecture, currently ["ae", "vae"].
        dataloader (_type_): A PyTorch dataloader. Usually a train dataloader.
        optimizer (_type_): Optimizer for training.
        device (str): Hardware
        epoch (int, optional): For description use, start from 1. Defaults to None.
        total_epoch (int, optional): For description use. Defaults to None.
        others (dict, optional): Other arguments. Defaults to None.

    Returns:
        dict: Losses stored in dictionary.  
                examples, AE will be {"reconstruction": float} and 
                VAE will be {"reconstruction": float, "latent": float}  

    """
    if model_code in ["lagging-vae", "reslag-vae"] and others["aggresive"]:
        enc_iter = others["aggresive-steps"]
    else:
        enc_iter = 1
    model.train()
    num_batches = len(dataloader)
    for enc_count in range(1, enc_iter+1): # Aggresive training for lagging-vae
        epoch_loss_dict = copy(LOSS_DICT_FORMAT[model_code])
        carrying_metrics = {"sigma2": []}
        it = iter(dataloader)
        if model_code not in ["lagging-vae", "reslag-vae"]:
            desc = "Training: " if epoch is None else f"Training {epoch}: "
        else:
            desc = f"Training (aggresive {enc_count}/{enc_iter}): " if epoch is None\
                  else f"Training {epoch} (aggresive {enc_count}/{enc_iter}): "

        for idx in tqdm(range(0, num_batches-1), desc=desc):
            # Extract input from dataloader
            input = next(it)[0]
            optimizer.zero_grad()
            input = input.to(device)
            s = input.shape
            D = s[1] * s[2] * s[3] # Data space dimension

            # Run prediction and evaluation
            if model_code == "ae":
                prediction = model(input)
                input = unnormalize(input, others["dataset"])
                loss = reconstruction_loss(prediction, input)
                loss.backward()
                optimizer.step()
                epoch_loss_dict["reconstruction"] += loss.item()
            elif model_code == "vae":
                prediction = model(input)
                input = unnormalize(input, others["dataset"])
                rec_loss = reconstruction_loss(prediction, input)
                latent_loss = model.vencoder.kl
                loss = rec_loss + latent_loss
                loss.backward()
                optimizer.step()
                epoch_loss_dict["reconstruction"] += rec_loss.item()
                epoch_loss_dict["latent"] += latent_loss.item()
            elif model_code == "beta-vae":
                prediction = model(input)
                input = unnormalize(input, others["dataset"])
                rec_loss = reconstruction_loss(prediction, input)
                # Using perceptual loss.
                perc_coef = 0 if "perceptual" not in others else others["perceptual"]
                perc_loss = torch.tensor(0).to(DEVICE) if perc_coef == 0 \
                        else perceptual_loss(prediction, input)
                loss = (1-perc_coef) * rec_loss + perc_coef * perc_loss
                latent_loss = model.vencoder.kl
                # beta-vae with control capacity increase 
                if model.capacity is not None:
                    curr_capacity = model.capacity * epoch / total_epoch
                    loss = loss + model.beta * torch.abs(latent_loss - curr_capacity)
                else:                
                    loss = loss + model.beta * latent_loss
                loss.backward()
                optimizer.step()
                epoch_loss_dict["reconstruction"] += rec_loss.item()
                epoch_loss_dict["perceptual"] += perc_loss.item()
                epoch_loss_dict["latent"] += latent_loss.item()
                epoch_loss_dict["total"] += loss.item()
            elif model_code == "mmd-vae":
                train_z = model.vencoder(input)
                prediction = model.decoder(train_z)
                input = unnormalize(input, others["dataset"])
                rec_loss = reconstruction_loss(prediction, input)
                # Using perceptual loss.
                perc_coef = 0 if "perceptual" not in others else others["perceptual"]
                perc_loss = torch.tensor(0).to(DEVICE) if perc_coef == 0 \
                        else perceptual_loss(prediction, input)
                loss = (1-perc_coef) * rec_loss + perc_coef * perc_loss
                latent_loss = model.vencoder.kl
                # Sample minibatch from prior (standard normal)
                # then perform MMD loss calculation.
                # This is following implementation from 
                # https://github.com/pratikm141/MMD-Variational-Autoencoder-Pytorch-InfoVAE
                true_samples = torch.randn((input.shape[0],model.latent_dim)).to(device)
                # true_samples = Variable(true_samples).to(device)

                mmd_loss = compute_mmd(true_samples, train_z)
                loss = loss + (1 - model.alpha) * latent_loss \
                            + (model.alpha + model.l - 1) * mmd_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3) # Prevent weight overflow
                optimizer.step()
                epoch_loss_dict["reconstruction"] += rec_loss.item()
                epoch_loss_dict["perceptual"] += perc_loss.item()
                epoch_loss_dict["latent"] += latent_loss.item()
                epoch_loss_dict["mmd"] += mmd_loss.item()
                epoch_loss_dict["total"] += loss.item()
            elif model_code == "osigma-vae":
                prediction = model(input)
                input = unnormalize(input, others["dataset"])
                rec_loss = reconstruction_loss(prediction, input) # D*MSELoss
                if hasattr(model, "kl"):
                    latent_loss = model.kl # KL(reduction="sum")
                else:
                    latent_loss = model.vencoder.kl
                # Calculate variance by MLE
                sigma2_minibatch = mle_variance(input)
                carrying_metrics["sigma2"].append(sigma2_minibatch.item())
                # Below is D ln(sigma) + (D/2sigma^2) MSE + KL_divergence
                loss = (D/2) * torch.log(sigma2_minibatch)   \
                    + (1/(2*sigma2_minibatch)) * rec_loss  \
                    + latent_loss
                loss.backward()
                optimizer.step()
                epoch_loss_dict["reconstruction"] += rec_loss.item()
                epoch_loss_dict["latent"] += latent_loss.item()
                epoch_loss_dict["total"] += loss.item()
            elif model_code == "osigma-ladder":
                prediction = model(input)
                input = unnormalize(input, others["dataset"])
                # print(prediction.shape, input.shape)
                rec_loss = reconstruction_loss(prediction, input) # D*MSELoss
                if hasattr(model, "kl"):
                    latent_loss = model.kl # KL(reduction="sum")
                else:
                    latent_loss = model.vencoder.kl
                # Calculate variance by MLE
                sigma2_minibatch = mle_variance(input)
                carrying_metrics["sigma2"].append(sigma2_minibatch.item())
                # Below is D ln(sigma) + (D/2sigma^2) MSE + KL_divergence
                loss = (D/2) * torch.log(sigma2_minibatch)   \
                    + (1/(2*sigma2_minibatch)) * rec_loss  \
                    + latent_loss
                loss.backward()
                optimizer.step()
                epoch_loss_dict["reconstruction"] += rec_loss.item()
                epoch_loss_dict["latent"] += latent_loss.item()
                epoch_loss_dict["total"] += loss.item()
                # print(round(rec_loss.item(),4),round(latent_loss.item(),4),round(loss.item(),4) )
            elif model_code in ["lagging-vae", "reslag-vae"]:
                prediction = model(input)
                input = unnormalize(input, others["dataset"])
                rec_loss = reconstruction_loss(prediction, input) # D*MSELoss
                if hasattr(model, "kl"):
                    latent_loss = model.kl # KL(reduction="sum")
                else:
                    latent_loss = model.vencoder.kl
                # Calculate variance by MLE
                sigma2_minibatch = mle_variance(input)
                carrying_metrics["sigma2"].append(sigma2_minibatch.item())
                # Below is D ln(sigma) + (D/2sigma^2) MSE + KL_divergence
                loss = (D/2) * torch.log(sigma2_minibatch)   \
                    + (1/(2*sigma2_minibatch)) * rec_loss  \
                    + latent_loss
                loss.backward()
                if enc_count < enc_iter:
                    # Update encoder optimizer only
                    optimizer.step_enc()
                else:
                    # Update bth optimizers at last aggresive step.
                    optimizer.step()
                epoch_loss_dict["reconstruction"] += rec_loss.item()
                epoch_loss_dict["latent"] += latent_loss.item()
                epoch_loss_dict["total"] += loss.item()
            else:
                raise NotImplementedError(f"The model [{model_code}] is not implemented yet.")
    

    return {key: value/num_batches for key,value in epoch_loss_dict.items()}, carrying_metrics


def evaluate(model, model_code:str, dataloader, device:str, epoch:int=None, total_epoch:int=None,
             others: dict = None, carrying_metrics: dict = None) -> dict:
    """Unsupervised evaluation: Compare images to images

    Args:
        model (_type_): PyTorch model
        model_code (str): Model architecture
        dataloader (_type_): A PyTorch dataloader
        device (str): Hardware
        epoch (int, optional): For description use, start from 1. Defaults to None.
        total_epoch (int, optional): For description use. Defaults to None.
        others (dict, optional): Other arguments. Defaults to None.
        carrying_metrics (dict, optional): Metrics that have to be transferred
                        from training, currently used in osigma-vae. Defaults to None.


    Returns:
        dict: Losses stored in dictionary.
    """
    epoch_loss_dict = copy(LOSS_DICT_FORMAT[model_code])
    model.eval()
    num_batches = len(dataloader)
    it = iter(dataloader)
    desc = "Evaluate: " if epoch is None else f"Evaluate {epoch}: "
    with torch.no_grad():
        for idx in tqdm(range(0, num_batches-1), 
                desc=desc):
            input = next(it)[0]
            input = input.to(device)
            s = input.shape
            D = s[1] * s[2] * s[3] # Data space dimension
            # For osigma-vae, calculate data variance based on training variance
            training_variance = float(np.mean(carrying_metrics["sigma2"]))
            
            if model_code == "ae":
                prediction = model(input)
                input = unnormalize(input, others["dataset"])
                loss = reconstruction_loss(prediction, input)
                epoch_loss_dict["reconstruction"] += loss.item()
            elif model_code == "vae":
                prediction = model(input)
                input = unnormalize(input, others["dataset"])
                rec_loss = reconstruction_loss(prediction, input)
                latent_loss = model.vencoder.kl

                epoch_loss_dict["reconstruction"] += rec_loss.item()
                epoch_loss_dict["latent"] += latent_loss.item()
            elif model_code == "beta-vae":
                prediction = model(input)
                input = unnormalize(input, others["dataset"])
                rec_loss = reconstruction_loss(prediction, input)
                # Using perceptual loss.
                perc_coef = 0 if "perceptual" not in others else others["perceptual"]
                perc_loss = torch.tensor(0).to(DEVICE) if perc_coef == 0 \
                        else perceptual_loss(prediction, input)
                total_loss = (1-perc_coef) * rec_loss + perc_coef * perc_loss
                latent_loss = model.vencoder.kl
                if model.capacity is not None:
                    curr_capacity = model.capacity * epoch / total_epoch
                    total_loss = total_loss + model.beta * torch.abs(latent_loss - curr_capacity)
                else:                
                    total_loss = total_loss + model.beta * latent_loss

                epoch_loss_dict["reconstruction"] += rec_loss.item()
                epoch_loss_dict["perceptual"] += perc_loss.item()
                epoch_loss_dict["latent"] += latent_loss.item()
                epoch_loss_dict["total"] += total_loss.item()
            elif model_code == "mmd-vae":
                train_z = model.vencoder(input)
                prediction = model.decoder(train_z)
                input = unnormalize(input, others["dataset"])
                rec_loss = reconstruction_loss(prediction, input)
                # Using perceptual loss.# Using perceptual loss.
                perc_coef = 0 if "perceptual" not in others else others["perceptual"]
                perc_loss = torch.tensor(0).to(DEVICE) if perc_coef == 0 \
                        else perceptual_loss(prediction, input)
                loss = (1-perc_coef) * rec_loss + perc_coef * perc_loss
                latent_loss = model.vencoder.kl
                # Sample minibatch from prior (standard normal)
                # then perform MMD loss calculation.
                # This is following implementation from 
                # https://github.com/pratikm141/MMD-Variational-Autoencoder-Pytorch-InfoVAE
                true_samples = torch.randn((input.shape[0],model.latent_dim)).to(device)
                # true_samples = Variable(true_samples).to(device)

                mmd_loss = compute_mmd(true_samples, train_z)
                loss = loss + (1 - model.alpha) * latent_loss \
                            + (model.alpha + model.l - 1) * mmd_loss
                epoch_loss_dict["reconstruction"] += rec_loss.item()
                epoch_loss_dict["perceptual"] += perc_loss.item()
                epoch_loss_dict["latent"] += latent_loss.item()
                epoch_loss_dict["mmd"] += mmd_loss.item()
                epoch_loss_dict["total"] += loss.item()
            elif model_code == "osigma-vae":
                prediction = model(input)
                input = unnormalize(input, others["dataset"])
                rec_loss = reconstruction_loss(prediction, input) # D*MSELoss
                if hasattr(model, "kl"):
                    latent_loss = model.kl # KL(reduction="sum")
                else:
                    latent_loss = model.vencoder.kl
                # Below is D ln(sigma) + (D/2sigma^2) MSE + KL_divergence
                loss = (D/2) * torch.log(torch.tensor(training_variance)) \
                    + (1/(2*training_variance)) * rec_loss  \
                    + latent_loss
                epoch_loss_dict["reconstruction"] += rec_loss.item()
                epoch_loss_dict["latent"] += latent_loss.item()
                epoch_loss_dict["total"] += loss.item()
            elif model_code == "osigma-ladder":
                prediction = model(input)
                input = unnormalize(input, others["dataset"])
                rec_loss = reconstruction_loss(prediction, input) # D*MSELoss
                if hasattr(model, "kl"):
                    latent_loss = model.kl # KL(reduction="sum")
                else:
                    latent_loss = model.vencoder.kl
                # Below is D ln(sigma) + (D/2sigma^2) MSE + KL_divergence
                loss = (D/2) * torch.log(torch.tensor(training_variance)) \
                    + (1/(2*training_variance)) * rec_loss  \
                    + latent_loss
                epoch_loss_dict["reconstruction"] += rec_loss.item()
                epoch_loss_dict["latent"] += latent_loss.item()
                epoch_loss_dict["total"] += loss.item()
            elif model_code in ["lagging-vae", "reslag-vae"]:
                prediction = model(input)
                input = unnormalize(input, others["dataset"])
                rec_loss = reconstruction_loss(prediction, input) # D*MSELoss
                if hasattr(model, "kl"):
                    latent_loss = model.kl # KL(reduction="sum")
                else:
                    latent_loss = model.vencoder.kl
                # Calculate variance by MLE
                sigma2_minibatch = mle_variance(input)
                carrying_metrics["sigma2"].append(sigma2_minibatch.item())
                # Below is D ln(sigma) + (D/2sigma^2) MSE + KL_divergence
                loss = (D/2) * torch.log(sigma2_minibatch)   \
                    + (1/(2*sigma2_minibatch)) * rec_loss  \
                    + latent_loss
                epoch_loss_dict["reconstruction"] += rec_loss.item()
                epoch_loss_dict["latent"] += latent_loss.item()
                epoch_loss_dict["total"] += loss.item()

            
    return {key: value/num_batches for key,value in epoch_loss_dict.items()}

def get_dataset_from_name(model_name):
    # Make sure model_name already exist
    with open(os.path.join(OUTPUT_ROOT, model_name, "config.json"), "r") as f:
        dataset = json.load(f)["dataset"]
    
    return DATASETS.index(dataset)

def merge_comparison(dataset):
    from_root = os.path.join("./files/outputs/", dataset)
    ori_path = os.path.join(from_root, "original.png")
    assert os.path.isfile(ori_path), "Original images not found."
    paths = [ori_path]
    # Collect experiments reconstructions.
    for model_code in ALL_MODELS:
        variant_path = os.path.join(from_root, f"{model_code}.png")
        if not os.path.isfile(variant_path): 
            continue # Skip
        paths.append(variant_path)
    # Read image and make collage
    images = [torchvision.io.read_image(path) for path in paths]
    collage = torchvision.utils.make_grid(
        torch.stack(images), nrow=1, padding=2
    ).float()/255.
    
    torchvision.utils.save_image(collage, os.path.join(from_root, "collage.png"))