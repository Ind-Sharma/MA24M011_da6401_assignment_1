"""
Optimization Algorithms
Implements: SGD, Momentum, RMSprop
"""
import numpy as np


class SGD:
    def __init__(self,lr,weight_decay=0.0):
        self.lr = lr
        self.wd = weight_decay

    def update(self,layers):
        for layer in layers:
            grad_W = layer.grad_W+self.wd * layer.W
            grad_b = layer.grad_b
            layer.W = layer.W-self.lr*grad_W
            layer.b = layer.b-self.lr*grad_b


class Momentum:
    def __init__(self,lr,weight_decay=0.0,momentum=0.9):
        self.eta = lr           # eta = learning rate
        self.gamma = momentum   # gamma = momentum coefficient
        self.wd = weight_decay
        self.v_W = []
        self.v_b = []

    def update(self, layers):
        if len(self.v_W) == 0:
            for layer in layers:
                self.v_W.append(np.zeros_like(layer.W))
                self.v_b.append(np.zeros_like(layer.b))

        for i in range(len(layers)):
            layer = layers[i]
            nabla_W = layer.grad_W+self.wd*layer.W   # nabla_theta L
            nabla_b = layer.grad_b

            # v_{t+1} = gamma * v_t + eta * nabla_theta L
            v_t_W = self.v_W[i]
            v_t_b = self.v_b[i]
            self.v_W[i] = self.gamma*v_t_W+self.eta*nabla_W
            self.v_b[i] = self.gamma*v_t_b+self.eta*nabla_b

            # theta_{t+1} = theta_t - v_{t+1}
            layer.W = layer.W-self.v_W[i]
            layer.b = layer.b-self.v_b[i]


class RMSprop:
    def __init__(self,lr,weight_decay=0.0,beta=0.99,eps=1e-8):
        self.eta = lr           # eta = learning rate
        self.beta = beta        # beta = decay rate
        self.epsilon = eps      # epsilon = for numerical stability
        self.wd = weight_decay
        self.v_W = []           # v_t = exponentially weighted avg of g_t^2
        self.v_b = []

    def update(self,layers):
        if len(self.v_W) == 0:
            for layer in layers:
                self.v_W.append(np.zeros_like(layer.W))
                self.v_b.append(np.zeros_like(layer.b))

        for i in range(len(layers)):
            layer = layers[i]
            g_W = layer.grad_W+self.wd*layer.W   # g_t
            g_b = layer.grad_b

            # v_t = beta * v_{t-1} + (1 - beta) * g_t^2
            v_prev_W = self.v_W[i]
            v_prev_b = self.v_b[i]
            self.v_W[i] = self.beta*v_prev_W+(1-self.beta)*(g_W*g_W)
            self.v_b[i] = self.beta*v_prev_b+(1- self.beta) (g_b*g_b)

            # w_new = w_old - (eta / sqrt(v_t + epsilon)) * g_t
            layer.W = layer.W-(self.eta/(np.sqrt(self.v_W[i])+self.epsilon))*g_W
            layer.b = layer.b-(self.eta/(np.sqrt(self.v_b[i])+self.epsilon))*g_b
