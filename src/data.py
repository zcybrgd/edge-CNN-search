import torch
from torchvision import datasets, transforms

#real CIFAR-100 per-channel mean/std 
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_transforms(train: bool):
    if train:
        return transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),])
    return transforms.Compose([transforms.ToTensor(),transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),])


def get_dataloaders(cfg: dict):
    data_dir = cfg["data"]["data_dir"]
    batch_size = cfg["data"]["batch_size"]
    num_workers = cfg["data"]["num_workers"]
    train_set = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=get_transforms(train=True) )
    test_set = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=get_transforms(train=False))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,num_workers=num_workers, pin_memory=True, drop_last=True,)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,num_workers=num_workers, pin_memory=True,)
    return train_loader, test_loader