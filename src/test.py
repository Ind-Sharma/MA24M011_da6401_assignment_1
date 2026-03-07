import numpy as np
import argparse
import os
import sys

sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ann.neural_network import NeuralNetwork

best_config = argparse.Namespace(
    dataset="mnist",
    epochs=20,
    batch_size=64,
    loss="cross_entropy",
    optimizer="rmsprop",
    weight_decay=0.0,
    learning_rate=0.001,
    num_layers=3,
    hidden_size=[128,128,128],
    hidden_layers=[128,128,128],
    activation="relu",
    weight_init="xavier"
)

model = NeuralNetwork(best_config)

src = os.path.dirname(os.path.abspath(__file__))
weights = np.load(os.path.join(src,"best_model.npy"),allow_pickle=True).item()

model.set_weights(weights)

print("model loaded ok")
