import sys, argparse, numpy as np
sys.path.insert(0, 'src')
from ann.neural_network import NeuralNetwork

args = argparse.Namespace(hidden_size=[32], hidden_layers=[32], activation='relu',
    loss='cross_entropy', optimizer='sgd', learning_rate=0.01, weight_decay=0.0,
    weight_init='xavier')
model = NeuralNetwork(args)

X = np.random.randn(100, 784)
y = np.array(list(range(10)) * 10)

for ep in range(20):
    loss = model.train(X, y, epochs=1, batch_size=10)
    result = model.evaluate(X, y)
    print(f"epoch {ep+1}: loss={loss:.4f} acc={result['accuracy']:.4f}")
