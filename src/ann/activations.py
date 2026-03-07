"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""
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
    return np.maximum(0, z)


def relu_grad(z):
    return (z > 0).astype(float)


def softmax(z):
    shifted = z - np.max(z,axis=0,keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z,axis=0,keepdims=True)


def softmax_grad(z):
    return 1.0
