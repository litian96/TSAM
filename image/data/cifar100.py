import numpy as np

import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utility.cutout import Cutout

class Cifar100:
    def __init__(self, self, batch_size, threads, corruption):
        mean, std = self._get_statistics()

        train_transform = transforms.Compose([
            torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            Cutout()
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_set = CIFAR100RandomLabels(root='./data', train=True, download=True, transform=train_transform, num_classes=100, corrupt_prob=corruption)
        test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)

        self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=threads)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def _get_statistics(self):
        train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transforms.ToTensor())
        data = torch.cat([d[0] for d in DataLoader(train_set, batch_size=512, shuffle=False)], dim=0)
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])


class CIFAR100RandomLabels(datasets.CIFAR100):
    def __init__(self, corrupt_prob=0.0, num_classes=100, **kwargs):
        super(CIFAR100RandomLabels, self).__init__(**kwargs)
        self.num_classes = num_classes
        if corrupt_prob > 0:
            self.corrupt_labels(corrupt_prob)

    def corrupt_labels(self, corrupt_prob):
        labels = np.array(self.targets)
        np.random.seed(12345)
        mask = np.random.rand(len(labels)) < corrupt_prob
        rnd_labels = np.random.choice(self.num_classes, mask.sum())
        labels[mask] = rnd_labels
        # we need to explicitly cast the labels from npy.int64 to int64,
        # otherwise pytorch will fail.
        labels = [int(x) for x in labels]
        self.targets = labels
