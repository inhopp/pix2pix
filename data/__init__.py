from .dataset import Dataset
import torch
import torchvision.transforms as transforms

def generate_loader(opt):
    dataset = Dataset
    img_size = opt.input_size
    
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    
    input_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    target_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    dataset = dataset(opt, input_transform=input_transform, target_transform=target_transform)
    
    kwargs = {
        "batch_size": opt.batch_size,
        "shuffle": True,
        "drop_last": True,
    }

    return torch.utils.data.DataLoader(dataset, **kwargs)