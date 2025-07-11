# FCSDM: Feature Causality-based Selection Dynamic Memory

This repository contains the implementation of **"Continual Unsupervised Generative Modeling via Feature Causality-based Selection Dynamic Memory"**, a novel approach for Online Task-Free Continual Learning (OTFCL) using diffusion models.

## Overview

Existing diffusion-based models have demonstrated impressive generative performance in large-scale supervised learning. However, their effectiveness in unsupervised continual learning is hindered by catastrophic forgetting. Our FCSDM method addresses this challenge by:

- **Adaptive Memory Clustering**: Forms memory clusters corresponding to different data categories
- **Causal Feature Selection**: Selectively retains samples critical for model training based on causality implications
- **Dynamic Cluster Pruning**: Prevents unbounded memory growth while maintaining performance

## Key Features

- **Feature Causality-based Selection**: Uses pre-trained image encoders to extract features and construct causal relationship graphs
- **Shared Sample Mechanism**: Allows samples to be stored across multiple clusters to enrich the causal graph
- **Dynamic Memory Management**: Balances memory efficiency with model performance
- **Task-Free Learning**: Operates without class labels or task boundaries

## Project Structure

```
FCSDM/
├── NetworkModels/           # Core network architectures
│   ├── MemoryUnitFramework_.py       # Memory unit framework
│   ├── MemoryClusteringFramework_.py # Memory clustering implementation
│   ├── DynamicDiffusionMixture_.py   # Dynamic diffusion mixture model
│   └── VAE_Model_.py                 # VAE-based student model
├── datasets/                # Dataset loading and preprocessing
│   ├── Data_Loading.py      # General data loading utilities
│   ├── MyCIFAR10.py        # CIFAR-10 dataset handler
│   └── MNIST32.py          # MNIST dataset handler
├── Task_Split/             # Task splitting utilities
│   ├── TaskFree_Split.py   # Task-free data splitting
│   └── Task_utilizes.py    # Task utilities
├── improved_diffusion/     # Improved diffusion model implementation
├── models/                 # Various model architectures
└── FCSDM_*.py             # Main training scripts for different datasets
```

## Usage

### Training on Different Datasets

```bash
# CIFAR-10
python FCSDM_CIFAR10.py

# MNIST
python FCSDM_MNIST.py

# Fashion-MNIST
python FCSDM_Fashion.py

# SVHN
python FCSDM_SVHN.py

# CelebA-HQ
python FCSDM_CelebAHQ128.py
python FCSDM_CelebAHQ256.py
```

### Cross-Domain Generation

```bash
# CelebA to Chair
python FCSDM_CelebAToChair64.py

# CelebA to ImageNet
python FCSDM_CelebAToImageNet.py
```

## Key Components

1. **Memory Unit Framework**: Manages dynamic memory clusters with causal feature selection
2. **Diffusion Components**: Handles generative modeling using diffusion processes
3. **Causal Graph Construction**: Builds relationships between features using pre-trained encoders
4. **Dynamic Cluster Pruning**: Maintains memory efficiency through intelligent sample replacement

## Requirements

- Python 3.7+
- PyTorch 1.8+
- torchvision
- numpy
- scikit-learn
- opencv-python
- clip-by-openai
- PIL

## Results

FCSDM achieves superior generative performance compared to state-of-the-art methods on OTFCL benchmarks, effectively addressing catastrophic forgetting in unsupervised continual learning scenarios.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{fcsdm2024,
  title={Continual Unsupervised Generative Modeling via Feature Causality-based Selection Dynamic Memory},
  author={Your Name},
  journal={Conference/Journal Name},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 