import numpy as np
from .activations import sigmoid, sigmoid_grad, relu, relu_grad


class NNLayer:
    def __init__(self,m,n,init="xavier"):
        self.b = np.zeros((m,1))
        if init == "random":
            self.W = np.random.randn(m,n)
        elif init == "zeros":
            self.W = np.zeros((m,n))
        else:
            self.W = np.random.randn(m,n) * np.sqrt(1.0/n)
        self.prev_input = None
        self.grad_W = None
        self.grad_b = None

    def forward_pass(self,input):
        self.prev_input = input
        return np.dot(self.W,input) + self.b

    def backward_pass(self,dZ):
        batch_size = dZ.shape[1]
        self.grad_W = (1/batch_size) * np.dot(dZ,self.prev_input.T)
        self.grad_b = (1/batch_size) * np.sum(dZ,axis=1,keepdims=True)
        return np.dot(self.W.T,dZ)


class ActivationLayer:
    def __init__(self,activation):
        self.activation = activation
        self.input = None

    def forward_pass(self, x):
        self.input = x
        if self.activation == "sigmoid":
            return sigmoid(x)
        if self.activation == "tanh":
            return np.tanh(x)
        return relu(x)

    def backward_pass(self, grad_from_next):
        if self.activation == "sigmoid":
            return grad_from_next * sigmoid_grad(self.input)
        if self.activation == "tanh":
            t = np.tanh(self.input)
            return grad_from_next * (1 - t * t)
        return grad_from_next * relu_grad(self.input)
