"""
Data loading utility for Quantum Neural Network experiments.

This module loads the MNIST dataset, keeps original 28x28 images for better accuracy,
filters selected digit classes for binary classification, and returns
PyTorch tensors ready for downstream use in quantum models.

Author: Harsh Gupta
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os
from typing import Tuple

def get_mnist_dataset(
    data_dir: str = "datasets/",
    image_size: Tuple[int, int] = (28, 28),  # Keep original MNIST size
    binary_digits: tuple = tuple(range(10)),
    train: bool = True,
    download: bool = True,
    limit_samples: int = None
) -> tuple:
    """
    Loads and preprocesses the MNIST dataset with optional class filtering.

    Args:
        data_dir (str): Directory where MNIST will be downloaded or read from.
        image_size (tuple): Image size in (H, W), default (28, 28) for no resizing.
        binary_digits (tuple): Tuple of allowed digits (e.g., (0, 1)).
        train (bool): If True, load training data; else load test data.
        download (bool): If True, download the dataset if not present.
        limit_samples (int): Optional limit on number of samples returned.

    Returns:
        images (torch.Tensor): Normalized images in shape [N, 1, H, W], ∈ [0, 1].
        labels (torch.Tensor): Corresponding digit labels in shape [N].
    """

    # Ensure dataset directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Define the preprocessing pipeline
    transform = transforms.Compose([
        transforms.ToTensor()  # Convert PIL image to tensor [0, 1]
    ])

    # Load MNIST dataset (training or test)
    mnist = datasets.MNIST(
        root=data_dir,
        train=train,
        transform=transform,
        download=download
    )

    # Filter dataset: retain only specified digit classes (e.g., 0 and 1)
    selected_indices = [
        idx for idx, label in enumerate(mnist.targets)
        if label.item() in binary_digits
    ]

    filtered_subset = Subset(mnist, selected_indices)

    # Optionally limit number of samples for quick prototyping
    if limit_samples is not None:
        filtered_subset = Subset(filtered_subset, list(range(min(limit_samples, len(filtered_subset)))))

    # Load all data into memory in one batch
    loader = DataLoader(
        dataset=filtered_subset,
        batch_size=len(filtered_subset),
        shuffle=True
    )

    # Return one batch: all filtered images
    images, labels = next(iter(loader))
    images = images.to(torch.float64)
    return images, labels