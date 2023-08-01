# This script provide some calculations to the numerics in main program.
from gradio_utils import *
from utils import dataset_loader_atomic
def get_meanstd(dataset):
    # CIFAR10, CIFAR100 and CelebA
    d = dataset_loader_atomic(dataset, "train")
    dl = DataLoader(d, batch_size=64, shuffle=False)
    dl = list(dl)
    n_channel = dl[0][0].shape[1]
    n = 0
    sum_first = torch.zeros((n_channel))
    sum_second = torch.zeros((n_channel))
    for i, single in enumerate(list(dl)):
        batch = single[0]
        n += batch.shape[0] * batch.shape[2] * batch.shape[3]
        sum_first += torch.sum(batch, dim=[0,2,3])
        sum_second += torch.sum(batch**2, dim=[0,2,3])
    mean = sum_first/n
    std = torch.sqrt((sum_second - n*mean**2)/(n-1))
    print(mean, std)

if __name__ == "__main__":
    get_meanstd("MNIST")
    # tensor([0.1307]) tensor([0.3081])
    get_meanstd("FashionMNIST")
    # tensor([0.2860]) tensor([0.3530])
    get_meanstd("CIFAR10")
    # tensor([0.4914, 0.4822, 0.4465]) tensor([0.2470, 0.2435, 0.2616])
    get_meanstd("CIFAR100")
    # tensor([0.5071, 0.4865, 0.4409]) tensor([0.2673, 0.2564, 0.2762])
    get_meanstd("CelebA")
    # tensor([0.5063, 0.4258, 0.3832]) tensor([0.3091, 0.2889, 0.2883])