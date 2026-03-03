"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""
import numpy as np


def sigmoid(z):
    denom = 1+np.exp(-z)
    return 1/denom


def sigmoid_grad(z):
    act_val = sigmoid(z)
    return act_val*(1-act_val)


def tanh(z):
    return np.tanh(z)


def tanh_grad(z):
    act_val = np.tanh(z)
    return 1-act_val*act_val


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
    exp_z = np.exp(z)
    sum_exp = np.sum(exp_z,axis=0)
    probs = exp_z/sum_exp
    return probs


def softmax_grad(z):
    return 1.0  # we are intentionally not using the gradient of softmax with cross entropy loss beacause it simplifies to 1.0 after the multiplication with the gradient from the final layer
