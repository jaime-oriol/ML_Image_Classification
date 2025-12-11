"""
Dataset loading and preparation for football logo classification.
Handles image transformations and creates train/val/test splits.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_transforms(augment=True):
    """
    Define image preprocessing transformations.

    Args:
        augment: If True, apply data augmentation (for training)
                 If False, only resize and normalize (for validation/test)

    Returns:
        Composed transformation pipeline
    """
    if augment:
        # Training transformations with light augmentation
        return transforms.Compose([
            # Resize all logos to 224x224 (standard input size for CNNs)
            transforms.Resize((224, 224)),

            # Randomly flip horizontally with 50% probability
            # Helps model learn that some logos look similar when flipped
            transforms.RandomHorizontalFlip(p=0.5),

            # Slightly vary brightness and contrast (20% range)
            # Makes model robust to different image qualities
            transforms.ColorJitter(brightness=0.2, contrast=0.2),

            # Convert PIL image to PyTorch tensor (0-1 range)
            transforms.ToTensor(),

            # Normalize using ImageNet statistics
            # These are standard values for pretrained models like ResNet
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # RGB channel means
                std=[0.229, 0.224, 0.225]     # RGB channel standard deviations
            )
        ])
    else:
        # Validation/test transformations without augmentation
        return transforms.Compose([
            # Same resize for consistency
            transforms.Resize((224, 224)),

            # Convert to tensor
            transforms.ToTensor(),

            # Same normalization for consistency with training
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def get_dataloaders(data_dir, batch_size=32, val_split=0.15, test_split=0.15):
    """
    Create train, validation, and test data loaders from logo dataset.

    Args:
        data_dir: Path to data folder containing league subfolders
        batch_size: Number of images per batch (32 is a good default)
        val_split: Fraction of data for validation (15% = 0.15)
        test_split: Fraction of data for testing (15% = 0.15)

    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        test_loader: DataLoader for testing
        class_names: List of league names (26 leagues)
    """
    # Load dataset using ImageFolder
    # Automatically assigns labels based on subfolder names (league names)
    full_dataset = datasets.ImageFolder(
        root=data_dir,
        transform=get_transforms(augment=False)  # Start with no augmentation
    )

    # Calculate split sizes
    total_size = len(full_dataset)  # Total number of logos (605)
    test_size = int(total_size * test_split)   # 15% for testing (~91 images)
    val_size = int(total_size * val_split)     # 15% for validation (~91 images)
    train_size = total_size - test_size - val_size  # Remaining 70% for training (~423 images)

    # Randomly split dataset into train/val/test
    # Use fixed seed (42) for reproducible splits
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # Fixed seed for reproducibility
    )

    # Apply augmentation ONLY to training data
    # This is important: we augment training but not validation/test
    train_dataset.dataset.transform = get_transforms(augment=True)

    # Create DataLoader for training
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,  # Process 32 images at once
        shuffle=True,           # Shuffle data every epoch (important for training)
        num_workers=2           # Use 2 parallel workers to load data faster
    )

    # Create DataLoader for validation
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,          # No need to shuffle validation data
        num_workers=2
    )

    # Create DataLoader for testing
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,          # No need to shuffle test data
        num_workers=2
    )

    # Return all three loaders plus the league names
    return train_loader, val_loader, test_loader, full_dataset.classes
