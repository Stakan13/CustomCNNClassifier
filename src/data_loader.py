import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from typing import Tuple, Dict


def get_transforms(image_size: Tuple[int, int] = (150, 150)) -> Dict:
    transform_train = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    return {
        'train': transform_train,
        'test': transform_test
    }


def get_data_loaders(
    train_dir: Path,
    test_dir: Path,
    batch_size: int = 64,
    image_size: Tuple[int, int] = (150, 150),
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader, Dict]:
    transforms_dict = get_transforms(image_size)
    
    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=transforms_dict['train']
    )
    
    test_dataset = datasets.ImageFolder(
        root=test_dir,
        transform=transforms_dict['test']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    class_names = train_dataset.classes
    
    return train_loader, test_loader, class_names


def analyze_dataset(train_dir: Path) -> Dict:
    from collections import Counter
    
    class_counts = Counter()
    
    for class_dir in train_dir.iterdir():
        if class_dir.is_dir():
            count = len(list(class_dir.glob('*.jpg')))
            class_counts[class_dir.name] = count
    
    return {
        'class_counts': dict(class_counts),
        'total_images': sum(class_counts.values()),
        'num_classes': len(class_counts),
        'min_per_class': min(class_counts.values()),
        'max_per_class': max(class_counts.values())
    }
