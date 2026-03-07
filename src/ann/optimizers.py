import numpy as np

class SGD:
    def __init__(self,lr,weight_decay=0.0):
        self.lr = lr
        self.wd = weight_decay

    def update(self,layers):
        for layer in layers:
            gw = layer.grad_W + self.wd*layer.W
            layer.W = layer.W - self.lr*gw
            layer.b = layer.b - self.lr*layer.grad_b

class Momentum:
    def __init__(self,lr,weight_decay=0.0,momentum=0.9):
        self.eta = lr
        self.gamma = momentum
        self.wd = weight_decay
        self.v_W = None
        self.v_b = None

    def update(self,layers):
        if self.v_W is None:
            self.v_W = [np.zeros_like(l.W) for l in layers]
            self.v_b = [np.zeros_like(l.b) for l in layers]
        for i,layer in enumerate(layers):
            self.v_W[i] = self.gamma*self.v_W[i]+self.eta*(layer.grad_W+self.wd*layer.W)
            self.v_b[i] = self.gamma*self.v_b[i]+self.eta*layer.grad_b
            layer.W = layer.W - self.v_W[i]
            layer.b = layer.b - self.v_b[i]

class RMSprop:
    def __init__(self,lr,weight_decay=0.0,beta=0.99,eps=1e-8):
        self.eta = lr
        self.beta = beta
        self.epsilon = eps
        self.wd = weight_decay
        self.v_W = None
        self.v_b = None

    def update(self,layers):
        if self.v_W is None:
            self.v_W = [np.zeros_like(l.W) for l in layers]
            self.v_b = [np.zeros_like(l.b) for l in layers]
        for i,layer in enumerate(layers):
            gw = layer.grad_W+self.wd*layer.W
            self.v_W[i] = self.beta*self.v_W[i]+(1-self.beta)*(gw*gw)
            self.v_b[i] = self.beta*self.v_b[i]+(1-self.beta)*(layer.grad_b*layer.grad_b)
            layer.W = layer.W - (self.eta/np.sqrt(self.v_W[i]+self.epsilon))*gw
            layer.b = layer.b - (self.eta/np.sqrt(self.v_b[i]+self.epsilon))*layer.grad_b
