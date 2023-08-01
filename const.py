# Constants, contain all variables to be used in any other scripts
# A centralized version of variable storage. 
# For example, to use device="cuda" in different script, just import it from here
# no need to initialize it in every script, which could lead to incompatibility. 

from torchvision import transforms
import torch
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False
# Amendable 
TEST_RUN = False # If True, use less data to train, speed up experiment iterations and finding hyperparameters.
TEST_RUN_TAKE = 1024
if TEST_RUN: print("===This is a test run, training will be blazingly fast due to using far less dataset. "\
                    + "Enable this when you want to test code functionalities, not accuracy.===")
TRAINING_NORMALIZE = False
KAIMING_INIT = False
# Fixed configuration
OUTPUT_ROOT = "./models/"
CACHE_ROOT = "./model-cache/"
CACHE_FILE = os.path.join(CACHE_ROOT, "recent-config.json")
CURRENT_MODEL_NAME = None
CURRENT_MODEL = None
CURRENT_DATASET_NAME = None
CURRENT_TRAIN_OR_VALID = None
CURRENT_DATASET = []
CURRENT_GRAYSCALE = None
HEADLESS_COMMAND = ""
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')      
CUTOFF_LR = 5e-5
MAX_NUM_DATAFRAMES = 300
DATASETS = ["MNIST", "FashionMNIST", "CIFAR10", "CIFAR100", "CelebA"]
OUTPUT_ROOT = "models"
CLASS_MAPPING = {
    ("ae", "MNIST"): ["Encoder", "Decoder"],
    ("ae", "FashionMNIST"): ["Encoder", "Decoder"],
    ("ae", "CIFAR10"): ["EncoderCIFAR", "DecoderCIFAR"],
    ("ae", "CIFAR100"): ["EncoderCIFAR", "DecoderCIFAR"],
    ("vae", "MNIST"): ["VariationalEncoder", "Decoder"],
    ("vae", "FashionMNIST"): ["VariationalEncoder", "Decoder"],
    ("vae", "CIFAR10"): ["VariationalEncoderCIFAR", "DecoderCIFAR"],
    ("vae", "CIFAR100"): ["VariationalEncoderCIFAR", "DecoderCIFAR"],
    ("beta-vae", "MNIST"): ["betaVariationalEncoder", "betaDecoder"],
    ("beta-vae", "FashionMNIST"): ["betaVariationalEncoder", "betaDecoder"],
    ("beta-vae", "CIFAR10"): ["betaVariationalEncoderCIFAR", "betaDecoderCIFAR"],
    ("beta-vae", "CIFAR100"): ["betaVariationalEncoderCIFAR", "betaDecoderCIFAR"],
    ("beta-vae", "CelebA"): ["betaVariationalEncoderCelebA", "betaDecoderCelebA"],
    ("mmd-vae", "MNIST"): ["MMDVariationalEncoder", "MMDDecoder"],
    ("mmd-vae", "FashionMNIST"): ["MMDVariationalEncoder", "MMDDecoder"],
    ("mmd-vae", "CIFAR10"): ["MMDVariationalEncoderCIFAR", "MMDDecoderCIFAR"],
    ("mmd-vae", "CIFAR100"): ["MMDVariationalEncoderCIFAR", "MMDDecoderCIFAR"],
    ("mmd-vae", "CelebA"): ["MMDVariationalEncoderCelebA", "MMDDecoderCelebA"],
    ("osigma-vae", "MNIST"): ["VariationalEncoder", "Decoder"],
    ("osigma-vae", "FashionMNIST"): ["VariationalEncoder", "Decoder"],
    ("osigma-vae", "CIFAR10"): ["VariationalEncoderCIFAR", "DecoderCIFAR"],
    ("osigma-vae", "CIFAR100"): ["VariationalEncoderCIFAR", "DecoderCIFAR"],
    ("osigma-vae", "CelebA"): ["VariationalEncoderCelebA", "DecoderCelebA"],
    ("osigma-ladder", "MNIST"): ["LadderVariationalEncoder", "LadderDecoderMNIST"],
    ("osigma-ladder", "FashionMNIST"): ["LadderVariationalEncoder", "LadderDecoderMNIST"],
    ("osigma-ladder", "CIFAR10"): ["LadderVariationalEncoder", "LadderDecoderCIFAR"],
    ("osigma-ladder", "CIFAR100"): ["LadderVariationalEncoder", "LadderDecoderCIFAR"],
    ("lagging-vae", "MNIST"): ["laggingVariationalEncoder", "laggingDecoder"],
    ("lagging-vae", "FashionMNIST"): ["laggingVariationalEncoder", "laggingDecoder"],
    ("lagging-vae", "CIFAR10"): ["laggingVariationalEncoderCIFAR", "laggingDecoderCIFAR"],
    ("lagging-vae", "CIFAR100"): ["laggingVariationalEncoderCIFAR", "laggingDecoderCIFAR"],
    ("lagging-vae", "CelebA"): ["laggingVariationalEncoderCelebA", "laggingDecoderCelebA"],
    ("reslag-vae", "CIFAR10"): ["ResLagVariationalEncoderCIFAR", "ResLagDecoderCIFAR"],
    ("reslag-vae", "CIFAR100"): ["ResLagVariationalEncoderCIFAR", "ResLagDecoderCIFAR"],
    ("reslag-vae", "CelebA"): ["ResLagVariationalEncoderCelebA", "ResLagDecoderCelebA"],
}
MODEL_TYPE_MAPPING = {
    "AutoEncoder": "ae",
    "Variational Autoencoder": "vae",
    "Beta VAE": "beta-vae",
    "MMD InfoVAE": "mmd-vae",
    "Optimal Sigma VAE": "osigma-vae",
    "Optimal Sigma Ladder VAE": "osigma-ladder",
    "Lagging/Aggresive VAE": "lagging-vae",
    "Residual Lagging VAE": "reslag-vae"
}
MODEL_TYPE_INV_MAPPING = {value: key for key, value in MODEL_TYPE_MAPPING.items()}
COMPATIBLE_DATAMAP = {
    "MNIST": ["MNIST", "FashionMNIST"],
    "FashionMNIST": ["MNIST", "FashionMNIST"],
    "CIFAR10": ["CIFAR10", "CIFAR100"],
    "CIFAR100": ["CIFAR10", "CIFAR100"],
    "CelebA": ["CelebA"],
}
MEAN_STD = {
    "MNIST": [(0.1307), (0.3081)],
    "FashionMNIST": [(0.2860), (0.3530)],
    "CIFAR10": [(0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)],
    "CIFAR100": [(0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)],
    "CelebA": [(0.5063, 0.4258, 0.3832), (0.3091, 0.2889, 0.2883)],
}
NORMALIZING_TRANSFORMS = {
    "MNIST": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN_STD["MNIST"][0], std=MEAN_STD["MNIST"][1]),
            ]
        ),
    "FashionMNIST": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN_STD["FashionMNIST"][0], std=MEAN_STD["FashionMNIST"][1]),
            ]
        ),
    "CIFAR10": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN_STD["CIFAR10"][0], std=MEAN_STD["CIFAR10"][1]),
            ]
        ),
    "CIFAR100": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN_STD["CIFAR100"][0], std=MEAN_STD["CIFAR100"][1]),
            ]
        ),
    "CelebA": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((64, 64)),
                transforms.Normalize(mean=MEAN_STD["CelebA"][0], std=MEAN_STD["CelebA"][1]),
            ]
        ),
}
LOSS_DICT_FORMAT = {
    "ae": {"reconstruction": 0, "perceptual": 0},
    "vae": {"reconstruction": 0, "perceptual": 0, "latent": 0},
    "beta-vae": {"reconstruction": 0, "perceptual": 0, "latent": 0, "total": 0},
    "mmd-vae": {"reconstruction": 0, "perceptual": 0, "mmd": 0, "latent": 0, "total": 0},
    "osigma-vae": {"reconstruction": 0, "perceptual": 0, "latent": 0, "total": 0},
    "osigma-ladder": {"reconstruction": 0, "perceptual": 0, "latent": 0, "total": 0},
    "lagging-vae": {"reconstruction": 0, "perceptual": 0, "latent": 0, "total": 0},
    "reslag-vae": {"reconstruction": 0, "perceptual": 0, "latent": 0, "total": 0},
}

ALL_MODELS = list(LOSS_DICT_FORMAT.keys())

if IN_COLAB:
    CELEBA_DATAROOT = "/content/data/"
    CELEBA_DOWNLOAD = False
else:
    CELEBA_DATAROOT = "./"
    CELEBA_DOWNLOAD = True