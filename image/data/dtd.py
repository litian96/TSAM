import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class DTD:
    def __init__(self, batch_size, threads):
        
        train_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])

        test_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])

        train_set = torchvision.datasets.DTD(root='./data', split='train', download=True, transform=train_transform)
        test_set = torchvision.datasets.DTD(root='./data', split='test', download=True, transform=test_transform)

        self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=threads)

