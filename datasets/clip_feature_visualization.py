#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import torch
import clip
from Data_Loading import *
from cv2_imageProcess import GetImage_cv
import warnings
warnings.filterwarnings('ignore')

# Set data root directory
DATA_ROOT = '/users/rnt529/scratch/DCM/data'

# Create output directory
os.makedirs('clip_feature_plots', exist_ok=True)

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

def extract_clip_features(images, device):
    """Extract features using CLIP model"""
    # Load CLIP model
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Process images in batches
    batch_size = 64
    features = []
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        
        # Convert images to format expected by CLIP
        if isinstance(batch, np.ndarray):
            # Handle numpy arrays
            if batch.shape[-1] == 1 or (len(batch.shape) == 4 and batch.shape[1] == 1):  # Grayscale
                # Convert grayscale to RGB by repeating channels
                if len(batch.shape) == 4 and batch.shape[1] == 1:  # (N,1,H,W) format
                    batch = np.repeat(batch, 3, axis=1)
                elif len(batch.shape) == 4 and batch.shape[3] == 1:  # (N,H,W,1) format
                    batch = np.repeat(batch, 3, axis=3)
                else:
                    # Handle unusual formats by reshaping if possible
                    try:
                        if len(batch.shape) == 3 and batch.shape[0] == 1:  # (1,H,W) format
                            batch = np.repeat(batch, 3, axis=0)
                            batch = np.transpose(batch, (1, 2, 0))  # Convert to (H,W,3)
                            batch = batch[np.newaxis, ...]  # Add batch dimension (1,H,W,3)
                        else:
                            print(f"Warning: Unusual image shape: {batch.shape}, attempting to convert")
                            # Try to reshape to standard format
                            if len(batch.shape) == 3:  # (N,H,W) or similar
                                batch = batch.reshape(batch.shape[0], batch.shape[1], batch.shape[2], 1)
                                batch = np.repeat(batch, 3, axis=3)
                    except Exception as e:
                        print(f"Failed to reshape image with shape {batch.shape}: {e}")
                        continue
            
            # Ensure values are in [0, 255] range
            if batch.max() <= 1.0:
                batch = (batch * 255).astype(np.uint8)
            
            # Handle different image formats
            try:
                # Preprocess images
                processed_images = []
                for img in batch:
                    # Ensure image has correct shape (H,W,3)
                    if len(img.shape) == 2:  # (H,W)
                        img = np.stack([img, img, img], axis=2)  # Convert to (H,W,3)
                    elif len(img.shape) == 3 and img.shape[0] == 3:  # (3,H,W)
                        img = np.transpose(img, (1, 2, 0))  # Convert to (H,W,3)
                    elif len(img.shape) == 3 and img.shape[2] != 3:
                        if img.shape[2] == 1:  # (H,W,1)
                            img = np.repeat(img, 3, axis=2)  # Convert to (H,W,3)
                        else:
                            print(f"Warning: Unusual image shape: {img.shape}, skipping")
                            continue
                    
                    # Convert to PIL Image and preprocess
                    try:
                        pil_img = Image.fromarray(img)
                        processed_images.append(preprocess(pil_img))
                    except Exception as e:
                        print(f"Error preprocessing image with shape {img.shape}: {e}")
                        continue
                
                if not processed_images:
                    print(f"Warning: No images could be processed in this batch")
                    continue
                
                processed_images = torch.stack(processed_images)
            except Exception as e:
                print(f"Error processing batch with shape {batch.shape}: {e}")
                continue
        else:
            # Handle torch tensors
            try:
                if batch.shape[1] == 1:  # Grayscale (N,1,H,W)
                    # Convert grayscale to RGB by repeating channels
                    batch = batch.repeat(1, 3, 1, 1)
                
                # Ensure values are in [0, 1] range
                if batch.max() > 1.0:
                    batch = batch / 255.0
                    
                # Preprocess images
                processed_images = torch.stack([preprocess(transforms.ToPILImage()(img)) for img in batch])
            except Exception as e:
                print(f"Error processing tensor batch with shape {batch.shape}: {e}")
                continue
        
        # Extract features
        try:
            with torch.no_grad():
                batch_features = model.encode_image(processed_images.to(device))
                features.append(batch_features.cpu())
        except Exception as e:
            print(f"Error extracting features: {e}")
            continue
    
    # Check if we have any features
    if not features:
        raise ValueError("No features could be extracted from the provided images")
    
    # Concatenate all features
    features = torch.cat(features, dim=0).numpy()
    return features

def create_tsne_visualization(features, title, filename_prefix, class_labels=None, class_names=None):
    """Create 3D t-SNE visualization of CLIP features"""
    print(f"Creating 3D t-SNE visualization for {title} CLIP features...")
    
    # Apply t-SNE dimensionality reduction
    reducer = TSNE(n_components=3, perplexity=min(30, len(features)-1), 
                  learning_rate='auto', init='pca', random_state=42)
    embedding = reducer.fit_transform(features)
    
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
    ax.view_init(elev=30, azim=45)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join('clip_feature_plots', f"{filename_prefix}_clip_tsne.png"), 
               dpi=300, bbox_inches='tight')
    
    plt.close()
    print(f"  CLIP feature visualization saved for {title}")
    
    return embedding

def analyze_dataset_with_clip(load_func, title, filename_prefix, max_samples=1000):
    """Analyze dataset using CLIP features and create t-SNE visualization"""
    try:
        print(f"Loading {title} dataset...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if title in ["MNIST", "Fashion-MNIST"]:
            # For MNIST-type datasets
            if title == "MNIST":
                data, labels = load_mnist("mnist")
                class_names = [str(i) for i in range(10)]  # Digit names 0-9
            else:
                data, labels = load_mnist("Fashion")
                class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
            
            # Limit sample count
            if len(data) > max_samples:
                indices = np.random.choice(len(data), max_samples, replace=False)
                data = data[indices]
                labels = labels[indices]
            
            # Extract CLIP features
            features = extract_clip_features(data, device)
            
            # Create t-SNE visualization
            create_tsne_visualization(features, title, filename_prefix, class_labels=labels, class_names=class_names)
        
        elif title == "CIFAR-10":
            # Special handling for CIFAR-10
            data, labels, class_names = load_cifar10()
            
            # Limit sample count if needed
            if len(data) > max_samples:
                indices = np.random.choice(len(data), max_samples, replace=False)
                data = data[indices]
                labels = labels[indices]
                
            # Extract CLIP features
            features = extract_clip_features(data, device)
            
            # Create t-SNE visualization with labels
            create_tsne_visualization(features, title, filename_prefix, class_labels=labels, class_names=class_names)
            
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
                                # Extract CLIP features
                                features = extract_clip_features(data, device)
                                # Create t-SNE visualization
                                create_tsne_visualization(features, title, filename_prefix)
                            else:
                                print(f"  Could not load any images for {title}")
                        else:
                            # These are actual image data
                            data = train_data[:min(max_samples, len(train_data))]
                            # Extract CLIP features
                            features = extract_clip_features(data, device)
                            # Create t-SNE visualization
                            create_tsne_visualization(features, title, filename_prefix)
                    elif len(result) == 4:
                        # Likely data and labels
                        train_data, train_labels, test_data, test_labels = result
                        data = train_data[:min(max_samples, len(train_data))]
                        labels = train_labels[:min(max_samples, len(train_labels))]
                        # Extract CLIP features
                        features = extract_clip_features(data, device)
                        # Create t-SNE visualization
                        create_tsne_visualization(features, title, filename_prefix, class_labels=labels)
                    else:
                        # Assume first element is the data
                        data = result[0]
                        if isinstance(data, (list, np.ndarray)) and len(data) > max_samples:
                            data = data[:max_samples]
                        # Extract CLIP features
                        features = extract_clip_features(data, device)
                        # Create t-SNE visualization
                        create_tsne_visualization(features, title, filename_prefix)
                else:
                    data = result
                    if isinstance(data, (list, np.ndarray)) and len(data) > max_samples:
                        data = data[:max_samples]
                    # Extract CLIP features
                    features = extract_clip_features(data, device)
                    # Create t-SNE visualization
                    create_tsne_visualization(features, title, filename_prefix)
            except Exception as e:
                print(f"  Error analyzing {title}: {e}")
    except Exception as e:
        print(f"Error analyzing {title} dataset: {e}")

def analyze_svhn_dataset_with_clip():
    """Specifically analyze SVHN dataset with CLIP features"""
    try:
        print("Loading SVHN dataset...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
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
        
        # Extract CLIP features
        features = extract_clip_features(data, device)
        
        # Create t-SNE visualization
        create_tsne_visualization(features, "SVHN", "svhn", class_labels=labels, class_names=class_names)
    except Exception as e:
        print(f"Error analyzing SVHN dataset: {e}")

# Main execution
if __name__ == "__main__":
    print("Analyzing datasets and generating CLIP feature visualizations...")
    
    # Make sure we have the required packages
    try:
        import clip
        from PIL import Image
        import torchvision.transforms as transforms
    except ImportError:
        print("Error: This script requires the clip, PIL, and torchvision packages.")
        print("Please install them with: pip install clip pillow torchvision")
        sys.exit(1)
    
    # Analyze CIFAR-10 dataset (if available)
    try:
        analyze_dataset_with_clip(load_cifar10, "CIFAR-10", "cifar10")
    except Exception as e:
        print(f"Error with CIFAR-10 dataset: {e}")
    
    # Analyze CACD dataset (if available)
    try:
        analyze_dataset_with_clip(Load_CACD, "CACD", "cacd")
    except Exception as e:
        print(f"Error with CACD dataset: {e}")
    
    # Analyze MNIST datasets
    analyze_dataset_with_clip(lambda: load_mnist("mnist"), "MNIST", "mnist")
    analyze_dataset_with_clip(lambda: load_mnist("Fashion"), "Fashion-MNIST", "fashion_mnist")
    
    # Analyze SVHN dataset with special handling
    analyze_svhn_dataset_with_clip()
    
    # Analyze CelebA dataset (if available)
    try:
        analyze_dataset_with_clip(Load_CelebA, "CelebA", "celeba")
    except Exception as e:
        print(f"Error with CelebA dataset: {e}")
    
    # Analyze FFHQ dataset (if available)
    try:
        analyze_dataset_with_clip(Load_FFHQ, "FFHQ", "ffhq")
    except Exception as e:
        print(f"Error with FFHQ dataset: {e}")
        
    # Analyze 3D Chair dataset (if available)
    try:
        analyze_dataset_with_clip(Load_3DChair, "3D Chair", "3d_chair")
    except Exception as e:
        print(f"Error with 3D Chair dataset: {e}")
    
    # Analyze ImageNet dataset (if available)
    try:
        analyze_dataset_with_clip(Load_ImageNet, "ImageNet", "imagenet")
    except Exception as e:
        print(f"Error with ImageNet dataset: {e}")
    
    print("Analysis complete! Check the 'clip_feature_plots' directory for results.") 