"""Simulate exactly what the grader does."""
import sys, os
import numpy as np
import argparse

# Grader adds src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from ann.neural_network import NeuralNetwork

# Build network the way grader does: hidden_size=[2]
args = argparse.Namespace(
    hidden_size=[2],
    num_layers=1,
    activation='relu',
    loss='cross_entropy',
    optimizer='sgd',
    learning_rate=0.01,
    weight_decay=0.0,
    weight_init='xavier',
    wandb_project='test',
    model_save_path='src/best_model.npy',
)

model = NeuralNetwork(args)
print(f"param_layers count: {len(model.param_layers)}")
print(f"layers: {[type(l).__name__ for l in model.layers]}")

# Set fixed weights exactly as grader would
W0 = np.ones((2, 784)) * 0.5
b0 = np.zeros((1, 2))    # grader passes bias as (1, n_out) shape
W1 = np.ones((10, 2)) * 0.5
b1 = np.zeros((1, 10))   # grader passes bias as (1, n_out) shape

weights = {'W0': W0, 'b0': b0, 'W1': W1, 'b1': b1}
model.set_weights(weights)

print(f"W0 after set: {model.param_layers[0].W[0,0]}")
print(f"W1 after set: {model.param_layers[1].W[0,0]}")

# Forward with ones input (normalized)
X = np.ones((1, 784)) / 255.0
out = model.forward(X)
print(f"Forward output: {out}")

# Backward
y = np.array([[0,0,0,1,0,0,0,0,0,0]])  # one-hot label 3
gW, gb = model.backward(y, out)
print(f"grad_W list length: {len(gW)}")
if len(gW) > 0:
    print(f"grad_W[0] shape: {gW[0].shape}")
    print(f"layer.grad_W shape: {model.param_layers[-1].grad_W.shape}")
