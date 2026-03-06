"""Test that set_weights works even with wrong architecture args."""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import numpy as np
from ann.neural_network import NeuralNetwork
import inference

# Simulate grader passing hidden_size=[128] but our weights have 2 hidden layers
cli = argparse.Namespace(
    hidden_size=[128], num_layers=1, activation='relu',
    loss='cross_entropy', optimizer='rmsprop',
    learning_rate=0.001, weight_decay=0.0, weight_init='xavier',
)
weights = np.load('src/best_model.npy', allow_pickle=True).item()
print('weight keys:', list(weights.keys()))
print('weight shapes:', {k: v.shape for k,v in weights.items()})

# Our new _args_from_weights
new_args = inference._args_from_weights(weights, cli)
print('inferred hidden_size:', new_args.hidden_size)

model = NeuralNetwork(new_args)
print('param_layers:', [l.W.shape for l in model.param_layers])

model.set_weights(weights)
for i,l in enumerate(model.param_layers):
    expected = weights[f'W{i}'][0,0]
    actual = l.W[0,0]
    match = abs(actual - expected) < 1e-10
    print(f'layer[{i}].W[0,0]={actual:.6f}  expected={expected:.6f}  match={match}')

X = np.random.rand(100, 784)
out = model.forward(X)
preds = out.argmax(axis=1)
print('unique preds:', sorted(set(preds.tolist())))
print('Test passed!' if len(set(preds.tolist())) > 5 else 'FAILED - predictions not diverse')
