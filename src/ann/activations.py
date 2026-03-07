import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1 - s)


def tanh(z):
    return np.tanh(z)


def tanh_grad(z):
    t = np.tanh(z)
    return 1 - t * t


def relu(z):
    output = np.zeros_like(z)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            if z[i,j] > 0:
                output[i,j] = z[i,j]
    return output


def relu_grad(z):
    output = np.zeros_like(z)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            if z[i,j] > 0:
                output[i,j] = 1.0
    return output


def softmax(z):
    shifted = z - np.max(z,axis=0,keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z,axis=0,keepdims=True)


def softmax_grad(z):
    return 1.0
