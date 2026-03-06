"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""
import numpy as np
from keras.datasets import mnist, fashion_mnist
try:
    from src.ann import NNLayer, ActivationLayer
except ImportError:
    from ann import NNLayer, ActivationLayer


def load_dataset(dataset_name):
    if dataset_name == "mnist":
        (X_train,y_train),(X_test,y_test) = mnist.load_data()
    else:
        (X_train,y_train),(X_test,y_test) = fashion_mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
    return X_train,y_train,X_test,y_test


def one_hot_encode(y,num_classes=10):
    m = y.shape[0]
    Y = np.zeros((m,num_classes))
    for i in range(m):
        Y[i, y[i]] = 1
    return Y


def build_network(args):
    """Build list of layers from args (train) or config (inference)."""
    inp_dim = 784    
    out_dim = 10
    h = args.hidden_layers
    if type(h) == int:
        h = [h]
    weight_init = args.weight_init
    activation = args.activation

    layers = []
    prev_size = inp_dim
    for i in range(len(h)):
        curr_size = h[i]
        layers.append(NNLayer(curr_size,prev_size,init=weight_init))
        layers.append(ActivationLayer(activation))
        prev_size = curr_size
    layers.append(NNLayer(out_dim,prev_size,init=weight_init))
    return layers
