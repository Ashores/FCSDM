"""
Data utility functions for visualization scripts.
This file provides data loading utilities for the visualization tools.
"""

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
import os
import sys

# Add parent directory to path to import from other modules if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def Give_DataStream_Supervised(dataset_name):
    """
    Load dataset for visualization purposes.
    
    Args:
        dataset_name: Name of the dataset (mnist, cifar10, etc.)
        
    Returns:
        train_data: Training data
        train_labels: Training labels
        test_data: Test data
        test_labels: Test labels
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'mnist':
        return load_mnist_for_viz()
    elif dataset_name == 'cifar10':
        return load_cifar10_for_viz()
    elif dataset_name == 'fashion_mnist':
        return load_fashion_mnist_for_viz()
    else:
        print(f"Dataset {dataset_name} not supported. Using MNIST as default.")
        return load_mnist_for_viz()

def load_mnist_for_viz():
    """Load MNIST dataset for visualization."""
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
        transforms.Resize((32, 32)),  # Resize to 32x32 to match model expectations
    ])
    
    # Load datasets
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Extract data and labels
    train_data = []
    train_labels = []
    for data, label in train_dataset:
        # Convert to 3 channels if needed
        if data.shape[0] == 1:
            data = data.repeat(3, 1, 1)
        train_data.append(data.numpy())
        train_labels.append(label)
    
    test_data = []
    test_labels = []
    for data, label in test_dataset:
        # Convert to 3 channels if needed
        if data.shape[0] == 1:
            data = data.repeat(3, 1, 1)
        test_data.append(data.numpy())
        test_labels.append(label)
    
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)
    
    print(f"MNIST dataset loaded: train_data shape: {train_data.shape}, test_data shape: {test_data.shape}")
    
    return train_data, train_labels, test_data, test_labels

def load_cifar10_for_viz():
    """Load CIFAR-10 dataset for visualization."""
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
    ])
    
    # Load datasets
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Extract data and labels
    train_data = []
    train_labels = []
    for data, label in train_dataset:
        train_data.append(data.numpy())
        train_labels.append(label)
    
    test_data = []
    test_labels = []
    for data, label in test_dataset:
        test_data.append(data.numpy())
        test_labels.append(label)
    
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)
    
    print(f"CIFAR-10 dataset loaded: train_data shape: {train_data.shape}, test_data shape: {test_data.shape}")
    
    return train_data, train_labels, test_data, test_labels

def load_fashion_mnist_for_viz():
    """Load Fashion-MNIST dataset for visualization."""
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
        transforms.Resize((32, 32)),  # Resize to 32x32 to match model expectations
    ])
    
    # Load datasets
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    # Extract data and labels
    train_data = []
    train_labels = []
    for data, label in train_dataset:
        # Convert to 3 channels if needed
        if data.shape[0] == 1:
            data = data.repeat(3, 1, 1)
        train_data.append(data.numpy())
        train_labels.append(label)
    
    test_data = []
    test_labels = []
    for data, label in test_dataset:
        # Convert to 3 channels if needed
        if data.shape[0] == 1:
            data = data.repeat(3, 1, 1)
        test_data.append(data.numpy())
        test_labels.append(label)
    
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)
    
    print(f"Fashion-MNIST dataset loaded: train_data shape: {train_data.shape}, test_data shape: {test_data.shape}")
    
    return train_data, train_labels, test_data, test_labels

# Function to load a small subset of data for testing
def load_test_data_subset(dataset_name, num_samples=100):
    """Load a small subset of data for testing."""
    train_data, train_labels, test_data, test_labels = Give_DataStream_Supervised(dataset_name)
    
    # Take a subset
    test_data = test_data[:num_samples]
    test_labels = test_labels[:num_samples]
    
    return test_data, test_labels 