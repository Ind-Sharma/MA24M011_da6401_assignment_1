"""
Simulate EXACTLY what grader test 4 does:
1. Build model with grader's cli
2. set_weights from best_model.npy
3. Call evaluate_model directly on that model
"""
import sys, os, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ann.neural_network import NeuralNetwork
import inference

# Grader builds model with THEIR cli - try the most likely config
# From test 3 args: hidden_size=[128,128,128], activation=tanh
for hidden, activation in [
    ([128,128,128], 'tanh'),
    ([128,128], 'relu'),
    ([128], 'relu'),
]:
    print(f"\n=== grader hidden={hidden}, activation={activation} ===")
    
    cli = argparse.Namespace(
        hidden_size=hidden,
        hidden_layers=hidden,
        num_layers=len(hidden),
        activation=activation,
        loss='cross_entropy',
        optimizer='sgd',
        learning_rate=0.001,
        weight_decay=0.0,
        weight_init='xavier',
        wandb_project='test',
    )
    
    # Grader loads pretrained weights
    weights = inference.load_model('src/pretrained_model.npy')
    print(f"loaded weight keys: {list(weights.keys())}")
    
    # Grader builds model and sets weights
    model = NeuralNetwork(cli)
    print(f"model param_layers before set: {[l.W.shape for l in model.param_layers]}")
    
    model.set_weights(weights)
    print(f"model param_layers after set: {[l.W.shape for l in model.param_layers]}")
    
    # Check W0 matches
    w0_expected = float(weights['W0'].flat[0])
    w0_actual = float(model.param_layers[0].W.flat[0])
    print(f"W0[0,0] expected={w0_expected:.6f}, got={w0_actual:.6f}, match={abs(w0_expected-w0_actual)<1e-8}")
    
    # Grader calls evaluate_model directly with their X_test, y_test
    # Simulate with normalized MNIST-like data
    np.random.seed(42)
    X_test = np.random.rand(200, 784)
    y_test = np.tile(np.arange(10), 20)
    
    result = inference.evaluate_model(model, X_test, y_test)
    print(f"F1 on random (expect ~0.1 if model works): {result['f1']:.4f}")
    print(f"unique preds: {sorted(set(model.forward(X_test).argmax(axis=1).tolist()))}")

# Also test with real MNIST-like normalized data
print("\n=== Final test with best architecture ===")
from sklearn.datasets import fetch_openml
try:
    data = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
except TypeError:
    data = fetch_openml('mnist_784', version=1, as_frame=False)
X = data.data.astype(float) / 255.0
y = data.target.astype(int)
X_test_real, y_test_real = X[60000:], y[60000:]

for hidden in [[128,128,128], [128,128], [128]]:
    cli = argparse.Namespace(
        hidden_size=hidden, hidden_layers=hidden, num_layers=len(hidden),
        activation='relu', loss='cross_entropy', optimizer='sgd',
        learning_rate=0.001, weight_decay=0.0, weight_init='xavier',
    )
    weights = inference.load_model('src/pretrained_model.npy')
    model = NeuralNetwork(cli)
    model.set_weights(weights)
    result = inference.evaluate_model(model, X_test_real, y_test_real)
    print(f"hidden={hidden}: F1={result['f1']:.4f}, layers={[l.W.shape for l in model.param_layers]}")
