# Assignment 1: Multi-Layer Perceptron for Image Classification

## Overview

This assignment implements a neural network from scratch using only NumPy. It includes layers, activations, optimizers, and loss functions, trained on MNIST or Fashion-MNIST datasets.

## Links

- **Weights & Biases Report**: [Add your W&B report link here]
- **GitHub Repository**: [Add your GitHub repository link here]

## Usage

**Training:**
```bash
python src/train.py -d mnist -e 20 -b 64 -l cross_entropy -lr 0.01 -a relu -sz 128
```

**Inference:**
```bash
python src/inference.py -m src/best_model.npy -d mnist
```

Best model and config are saved to `src/best_model.npy` and `src/best_model_config.json` based on test F-1 score.

