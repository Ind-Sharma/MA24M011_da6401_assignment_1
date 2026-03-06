"""Test model.evaluate() after grader's set_weights round-trip."""
import sys, os, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ann.neural_network import NeuralNetwork
import inference

# Load MNIST
from sklearn.datasets import fetch_openml
try:
    data = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
except TypeError:
    data = fetch_openml('mnist_784', version=1, as_frame=False)
X = data.data.astype(float) / 255.0
y = data.target.astype(int)
X_test, y_test = X[60000:], y[60000:]

pretrained = np.load('src/pretrained_model.npy', allow_pickle=True).item()

for hidden in [[128,128,128], [128,128], [128]]:
    cli = argparse.Namespace(
        hidden_size=hidden, hidden_layers=hidden, num_layers=len(hidden),
        activation='relu', loss='cross_entropy', optimizer='sgd',
        learning_rate=0.001, weight_decay=0.0, weight_init='xavier',
    )
    model = NeuralNetwork(cli)
    model.set_weights(pretrained)
    best_weights = model.get_weights()
    np.save('test_grader_saved.npy', best_weights)
    
    # Grader's load_model (plain np.load)
    loaded = np.load('test_grader_saved.npy', allow_pickle=True).item()
    print(f"\nhidden={hidden}: saved keys={list(loaded.keys())}")
    
    # Grader rebuilds model with THEIR cli and sets these weights
    model2 = NeuralNetwork(cli)
    model2.set_weights(loaded)
    
    # Test both evaluate methods
    r1 = model2.evaluate(X_test, y_test)
    print(f"  model.evaluate() F1: {r1['f1']:.4f}")
    
    r2 = inference.evaluate_model(model2, X_test, y_test)
    print(f"  inference.evaluate_model() F1: {r2['f1']:.4f}")

import os; os.remove('test_grader_saved.npy')
