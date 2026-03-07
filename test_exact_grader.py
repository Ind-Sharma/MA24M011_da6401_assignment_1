"""
Exact simulation of what the grader does for test 4.
Tests all possible scenarios.
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import numpy as np

# === Grader's load_model (their exact code) ===
def grader_load_model(model_path):
    data = np.load(model_path, allow_pickle=True).item()
    return data

# Load test data (simulating TF MNIST - raw 0-255)
import tensorflow as tf
(Xtr, ytr), (Xte, yte) = tf.keras.datasets.mnist.load_data()
Xte_raw = Xte.reshape(-1, 784)          # shape (10000, 784), values 0-255
Xte_norm = Xte_raw / 255.0              # normalized

print(f"Xte_raw: shape={Xte_raw.shape}, max={Xte_raw.max()}, dtype={Xte_raw.dtype}")
print(f"Xte_norm: shape={Xte_norm.shape}, max={Xte_norm.max():.4f}, dtype={Xte_norm.dtype}")

from ann.neural_network import NeuralNetwork
import inference

# === SCENARIO 1: grader uses hidden=[128,128,128] tanh (test 3 args) ===
print("\n=== SCENARIO 1: grader cli = [128,128,128] tanh, X_test raw 0-255 ===")
cli = argparse.Namespace(hidden_size=[128,128,128], hidden_layers=[128,128,128], num_layers=3,
    activation='tanh', loss='cross_entropy', optimizer='sgd',
    learning_rate=0.001, weight_decay=0.0, weight_init='xavier')

# Step 1: grader loads from args.model_path
args = inference.parse_arguments()
print(f"args.model_path = {args.model_path}")
weights = grader_load_model(args.model_path)
print(f"loaded keys: {list(weights.keys())}")

# Step 2: grader does set_weights + get_weights + save
model = NeuralNetwork(cli)
model.set_weights(weights)
best_weights = model.get_weights()
np.save("test_best_model.npy", best_weights)

# Step 3: test 4 - grader loads saved model and calls evaluate_model
loaded_bw = grader_load_model("test_best_model.npy")
model2 = NeuralNetwork(cli)
model2.set_weights(loaded_bw)

print(f"model2._activation = {model2._activation}")
print(f"model2 param_layers: {[l.W.shape for l in model2.param_layers]}")

# Scenario 1a: grader passes raw X_test (0-255)
r1a = inference.evaluate_model(model2, Xte_raw, yte)
print(f"evaluate_model(raw): F1={r1a['f1']:.4f}")

# Scenario 1b: grader passes normalized X_test
r1b = inference.evaluate_model(model2, Xte_norm, yte)
print(f"evaluate_model(norm): F1={r1b['f1']:.4f}")

# Scenario 1c: grader calls model.evaluate() directly
r1c = model2.evaluate(Xte_raw, yte)
print(f"model.evaluate(raw): F1={r1c['f1']:.4f}")

r1d = model2.evaluate(Xte_norm, yte)
print(f"model.evaluate(norm): F1={r1d['f1']:.4f}")

# === SCENARIO 2: inference.main() is called ===
print("\n=== SCENARIO 2: inference.main() called ===")
result = inference.main()
print(f"inference.main() F1={result['f1']:.4f}")

import os; os.remove("test_best_model.npy")
