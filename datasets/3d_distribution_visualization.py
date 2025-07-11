#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import torch
from Data_Loading import *
from cv2_imageProcess import GetImage_cv
import warnings
warnings.filterwarnings('ignore')

# Set data root directory
DATA_ROOT = '/users/rnt529/scratch/DCM/data'

# Create output directory
os.makedirs('3d_distribution_plots', exist_ok=True)

# Patch functions to use correct data paths
original_load_mnist = load_mnist
def patched_load_mnist(dataset_name):
    if dataset_name in ["mnist", "Fashion", "MNIST"]:
        dataset_name = os.path.join(DATA_ROOT, dataset_name)
    return original_load_mnist(dataset_name)
load_mnist = patched_load_mnist

original_load_mnist_tanh = load_mnist_tanh
def patched_load_mnist_tanh(dataset_name):
    if dataset_name in ["mnist", "Fashion", "MNIST"]:
        dataset_name = os.path.join(DATA_ROOT, dataset_name)
    return original_load_mnist_tanh(dataset_name)
load_mnist_tanh = patched_load_mnist_tanh

# Function to load CIFAR-10 dataset
def load_cifar10():
    """Load CIFAR-10 dataset using torchvision if available"""
    try:
        import torchvision
        import torchvision.transforms as transforms
        
        # Define transform to normalize the data
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        # Download and load the training data
        trainset = torchvision.datasets.CIFAR10(root=os.path.join(DATA_ROOT, 'cifar10'), 
                                              train=True, download=True, transform=transform)
        
        # CIFAR-10 class names
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Extract data and labels
        data = []
        labels = []
        
        # Limit to 5000 samples for memory efficiency
        sample_count = min(5000, len(trainset))
        indices = np.random.choice(len(trainset), sample_count, replace=False)
        
        for idx in indices:
            image, label = trainset[idx]
            data.append(image)
            labels.append(label)
        
        # Convert to numpy arrays
        data = torch.stack(data)
        labels = np.array(labels)
        
        # Convert labels to one-hot encoding
        labels_one_hot = np.zeros((labels.size, 10))
        labels_one_hot[np.arange(labels.size), labels] = 1
        
        print(f"Loaded {len(data)} CIFAR-10 images with {len(np.unique(labels))} classes")
        return data, labels_one_hot, class_names
    
    except ImportError:
        print("torchvision is required to load CIFAR-10 dataset")
        raise

def create_tsne_visualization(data, title, filename_prefix, max_samples=1000, class_labels=None, class_names=None):
    """Create 3D visualization of dataset distribution using t-SNE"""
    print(f"Creating 3D t-SNE visualization for {title}...")
    
    # Convert to numpy array if it's a tensor
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    
    # Limit sample count
    if len(data) > max_samples:
        indices = np.random.choice(len(data), max_samples, replace=False)
        data = data[indices]
        if class_labels is not None:
            class_labels = class_labels[indices]
    
    # Handle different image formats
    if len(data.shape) == 4:
        if data.shape[1] == 3 or data.shape[1] == 1:  # (N, C, H, W)
            # Convert to (N, H, W, C) format
            data_vis = data.transpose(0, 2, 3, 1)
        else:  # (N, H, W, C)
            data_vis = data
    else:
        # Assume grayscale images
        data_vis = data
    
    # Flatten the data for dimension reduction
    flat_data = data_vis.reshape(data_vis.shape[0], -1)
    
    # Apply t-SNE dimensionality reduction to get 3D representation
    reducer = TSNE(n_components=3, perplexity=min(30, len(flat_data)-1), 
                  learning_rate='auto', init='pca', random_state=42)
    embedding = reducer.fit_transform(flat_data)
    
    # Create 3D visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Determine coloring
    if class_labels is not None:
        # If we have class labels, use them for coloring
        if len(class_labels.shape) > 1 and class_labels.shape[1] > 1:
            # One-hot encoded labels
            labels = np.argmax(class_labels, axis=1)
        else:
            # Direct labels
            labels = class_labels
        
        # Use different colors for each class
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        # Plot each class separately with same marker but different colors
        for i, label in enumerate(unique_labels):
            idx = labels == label
            color = colors[i % len(colors)]
            
            # Use class name if available, otherwise use class number
            if class_names is not None and int(label) < len(class_names):
                label_text = class_names[int(label)]
            else:
                label_text = f'Class {label}'
                
            ax.scatter(
                embedding[idx, 0], 
                embedding[idx, 1], 
                embedding[idx, 2],
                marker='o',
                color=color,
                s=50, 
                alpha=0.8,
                label=label_text
            )
        
        # Add legend
        ax.legend(title="Classes", loc="upper right")
    else:
        # Otherwise use sample index for coloring
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], 
                           c=range(len(embedding)), cmap='viridis', s=50, alpha=0.8)
        plt.colorbar(scatter, ax=ax, label="Sample Index")
    
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Enable interactive rotation
    ax.view_init(elev=30, azim=45)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join('3d_distribution_plots', f"{filename_prefix}_tsne_3d.png"), 
               dpi=300, bbox_inches='tight')
    
    plt.close()
    print(f"  t-SNE visualization saved for {title}")
    
    return embedding

def analyze_dataset(load_func, title, filename_prefix, max_samples=1000):
    """Analyze dataset and create t-SNE visualization"""
    try:
        print(f"Loading {title} dataset...")
        if title in ["MNIST", "Fashion-MNIST"]:
            # For MNIST-type datasets
            if title == "MNIST":
                data, labels = load_mnist("mnist")
                class_names = [str(i) for i in range(10)]  # Digit names 0-9
            else:
                data, labels = load_mnist("Fashion")
                class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
            
            # Create t-SNE visualization
            create_tsne_visualization(data, title, filename_prefix, max_samples=min(500, max_samples), 
                                     class_labels=labels, class_names=class_names)
        
        elif title == "CIFAR-10":
            # Special handling for CIFAR-10
            data, labels, class_names = load_cifar10()
            create_tsne_visualization(data, title, filename_prefix, max_samples=min(500, max_samples), 
                                     class_labels=labels, class_names=class_names)
            
        else:
            # For other datasets
            try:
                result = load_func()
                
                # Handle different return formats
                if isinstance(result, tuple):
                    if len(result) == 2:
                        train_data, test_data = result
                        # Check if these are file paths or actual image data
                        if isinstance(train_data, (list, np.ndarray)) and isinstance(train_data[0], str):
                            # These are file paths, load a sample
                            sample_paths = np.random.choice(train_data, min(max_samples, len(train_data)), replace=False)
                            loaded_images = []
                            for path in sample_paths:
                                try:
                                    img = GetImage_cv(path, input_height=64, input_width=64, 
                                                   resize_height=64, resize_width=64, crop=True)
                                    loaded_images.append(img)
                                except Exception as e:
                                    print(f"  Error loading {path}: {e}")
                            
                            if loaded_images:
                                data = np.array(loaded_images)
                                # Create t-SNE visualization
                                create_tsne_visualization(data, title, filename_prefix, max_samples=min(500, max_samples))
                            else:
                                print(f"  Could not load any images for {title}")
                        else:
                            # These are actual image data
                            data = train_data[:min(max_samples, len(train_data))]
                            # Create t-SNE visualization
                            create_tsne_visualization(data, title, filename_prefix, max_samples=min(500, max_samples))
                    elif len(result) == 4:
                        # Likely data and labels
                        train_data, train_labels, test_data, test_labels = result
                        data = train_data[:min(max_samples, len(train_data))]
                        labels = train_labels[:min(max_samples, len(train_labels))]
                        # Create t-SNE visualization
                        create_tsne_visualization(data, title, filename_prefix, max_samples=min(500, max_samples), class_labels=labels)
                    else:
                        # Assume first element is the data
                        data = result[0]
                        if isinstance(data, (list, np.ndarray)) and len(data) > max_samples:
                            data = data[:max_samples]
                        # Create t-SNE visualization
                        create_tsne_visualization(data, title, filename_prefix, max_samples=min(500, max_samples))
                else:
                    data = result
                    if isinstance(data, (list, np.ndarray)) and len(data) > max_samples:
                        data = data[:max_samples]
                    # Create t-SNE visualization
                    create_tsne_visualization(data, title, filename_prefix, max_samples=min(500, max_samples))
            except Exception as e:
                print(f"  Error analyzing {title}: {e}")
    except Exception as e:
        print(f"Error analyzing {title} dataset: {e}")

def analyze_svhn_dataset():
    """Specifically analyze SVHN dataset to handle the return format"""
    try:
        print("Loading SVHN dataset...")
        x_train, y_train, x_test, y_test = GetSVHN_DataSet()
        
        # Convert labels to one-hot encoding
        y_train_one_hot = np.eye(10)[y_train.reshape(-1).astype(int)]
        
        # Limit samples
        max_samples = 1000
        indices = np.random.choice(len(x_train), min(max_samples, len(x_train)), replace=False)
        data = x_train[indices]
        labels = y_train_one_hot[indices]
        
        # Class names for SVHN (digits 0-9)
        class_names = [str(i) for i in range(10)]
        
        # Create t-SNE visualization
        create_tsne_visualization(data, "SVHN", "svhn", max_samples=min(500, max_samples), 
                                 class_labels=labels, class_names=class_names)
    except Exception as e:
        print(f"Error analyzing SVHN dataset: {e}")

# Main execution
if __name__ == "__main__":
    print("Analyzing datasets and generating t-SNE visualizations...")

    # Analyze CIFAR-10 dataset (if available)
    try:
        analyze_dataset(load_cifar10, "CIFAR-10", "cifar10")
    except Exception as e:
        print(f"Error with CIFAR-10 dataset: {e}")
    
    try:
        analyze_dataset(Load_CACD, "CACD", "cacd")
    except Exception as e:
        print(f"Error with CACD dataset: {e}")
    # Analyze MNIST datasets
    analyze_dataset(lambda: load_mnist("mnist"), "MNIST", "mnist")
    analyze_dataset(lambda: load_mnist("Fashion"), "Fashion-MNIST", "fashion_mnist")
    

    
    # Analyze SVHN dataset with special handling
    analyze_svhn_dataset()
    
    # Analyze CelebA dataset (if available)
    try:
        analyze_dataset(Load_CelebA, "CelebA", "celeba")
    except Exception as e:
        print(f"Error with CelebA dataset: {e}")
    
    # Analyze CACD dataset (if available)
 
    
    # Analyze FFHQ dataset (if available)
    try:
        analyze_dataset(Load_FFHQ, "FFHQ", "ffhq")
    except Exception as e:
        print(f"Error with FFHQ dataset: {e}")
        
    # Analyze 3D Chair dataset (if available)
    try:
        analyze_dataset(Load_3DChair, "3D Chair", "3d_chair")
    except Exception as e:
        print(f"Error with 3D Chair dataset: {e}")
    
    # Analyze ImageNet dataset (if available)
    try:
        analyze_dataset(Load_ImageNet, "ImageNet", "imagenet")
    except Exception as e:
        print(f"Error with ImageNet dataset: {e}")
    
    print("Analysis complete! Check the '3d_distribution_plots' directory for results.") 