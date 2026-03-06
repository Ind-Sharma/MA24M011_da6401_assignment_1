"""Simulate exactly what grader does for inference test."""
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

import inference
import numpy as np
from ann.neural_network import NeuralNetwork

# Step 1: parse args (grader calls this)
args = inference.parse_arguments()
print("parsed args:", vars(args))

# Step 2: grader builds model from args
model = NeuralNetwork(args)
print("param_layers:", len(model.param_layers))
print("layer shapes:", [l.W.shape for l in model.param_layers])

# Step 3: grader loads weights and sets them
weights = inference.load_model(args.model_path)
print("weight keys:", list(weights.keys()))
model.set_weights(weights)
print("W0 after set:", model.param_layers[0].W[0,0])

# Step 4: quick forward test
X = np.random.randn(5, 784)
out = model.forward(X)
print("forward output shape:", out.shape)
print("argmax predictions:", np.argmax(out, axis=1))
print("all same?", len(set(np.argmax(out, axis=1).tolist())) == 1)
