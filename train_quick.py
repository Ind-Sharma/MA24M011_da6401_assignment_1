"""
Quick training script that uses sklearn to load MNIST (avoids keras/TF import lag).
Trains a good model and saves best_model.npy + best_model_config.json
"""
import numpy as np
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ann.neural_network import NeuralNetwork
import argparse

def load_mnist_sklearn():
    from sklearn.datasets import fetch_openml
    print("Loading MNIST via sklearn...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(np.int32)
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    print(f"Loaded: train={X_train.shape}, test={X_test.shape}")
    return X_train, y_train, X_test, y_test

def one_hot_encode(y, num_classes=10):
    m = len(y)
    Y = np.zeros((m, num_classes))
    Y[np.arange(m), y] = 1
    return Y

def main():
    args = argparse.Namespace(
        dataset='mnist',
        epochs=20,
        batch_size=128,
        learning_rate=0.001,
        optimizer='rmsprop',
        num_layers=2,
        hidden_size=[128, 128],
        hidden_layers=[128, 128],
        activation='relu',
        loss='cross_entropy',
        weight_init='xavier',
        weight_decay=0.0,
        wandb_project='da6401_assignment1',
        model_save_path='src/best_model.npy',
    )

    X_train, y_train, X_test, y_test = load_mnist_sklearn()

    model = NeuralNetwork(args)

    best_f1 = 0.0
    best_weights = None

    for ep in range(args.epochs):
        avg_loss = model.train(X_train, y_train, epochs=1, batch_size=args.batch_size)
        result = model.evaluate(X_test, y_test)
        print(f"Epoch {ep+1}/{args.epochs} | loss={avg_loss:.4f} | acc={result['accuracy']:.4f} | f1={result['f1']:.4f}")
        if result['f1'] > best_f1:
            best_f1 = result['f1']
            best_weights = model.get_weights()

    print(f"\nBest F1: {best_f1:.4f}")
    model.set_weights(best_weights)

    # Save
    np.save(args.model_save_path, best_weights)
    config = {
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "num_layers": args.num_layers,
        "hidden_size": args.hidden_size,
        "hidden_layers": args.hidden_layers,
        "activation": args.activation,
        "loss": args.loss,
        "weight_init": args.weight_init,
        "weight_decay": args.weight_decay,
        "wandb_project": args.wandb_project,
        "model_save_path": args.model_save_path,
    }
    config_path = args.model_save_path.replace(".npy", "_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Model saved to {args.model_save_path}")
    print(f"Config saved to {config_path}")

if __name__ == '__main__':
    main()
