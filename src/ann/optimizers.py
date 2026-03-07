import numpy as np


class SGD:
    def __init__(self,lr,weight_decay=0.0):
        self.lr = lr
        self.wd = weight_decay

    def update(self,layers):
        for layer in layers:
            grad_W = layer.grad_W + self.wd*layer.W
            grad_b = layer.grad_b
            layer.W = layer.W - self.lr*grad_W
            layer.b = layer.b - self.lr*grad_b


class Momentum:
    def __init__(self,lr,weight_decay=0.0,momentum=0.9):
        self.eta = lr
        self.gamma = momentum
        self.wd = weight_decay
        self.v_W = []
        self.v_b = []

    def update(self,layers):
        if len(self.v_W) == 0:
            for layer in layers:
                self.v_W.append(np.zeros_like(layer.W))
                self.v_b.append(np.zeros_like(layer.b))
        for i in range(len(layers)):
            layer = layers[i]
            nabla_W = layer.grad_W + self.wd*layer.W
            nabla_b = layer.grad_b
            self.v_W[i] = self.gamma*self.v_W[i] + self.eta*nabla_W
            self.v_b[i] = self.gamma*self.v_b[i] + self.eta*nabla_b
            layer.W = layer.W - self.v_W[i]
            layer.b = layer.b - self.v_b[i]


class RMSprop:
    def __init__(self,lr,weight_decay=0.0,beta=0.99,eps=1e-8):
        self.eta = lr
        self.beta = beta
        self.epsilon = eps
        self.wd = weight_decay
        self.v_W = []
        self.v_b = []

    def update(self,layers):
        if len(self.v_W) == 0:
            for layer in layers:
                self.v_W.append(np.zeros_like(layer.W))
                self.v_b.append(np.zeros_like(layer.b))
        for i in range(len(layers)):
            layer = layers[i]
            g_W = layer.grad_W + self.wd*layer.W
            g_b = layer.grad_b
            self.v_W[i] = self.beta*self.v_W[i] + (1-self.beta)*(g_W*g_W)
            self.v_b[i] = self.beta*self.v_b[i] + (1-self.beta)*(g_b*g_b)
            layer.W = layer.W - (self.eta/np.sqrt(self.v_W[i]+self.epsilon))*g_W
            layer.b = layer.b - (self.eta/np.sqrt(self.v_b[i]+self.epsilon))*g_b
