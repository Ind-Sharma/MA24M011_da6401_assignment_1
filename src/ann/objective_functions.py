import numpy as np


def mse_forward(y_hat,y):
    n = y.shape[1]
    diff = y_hat - y
    loss = (1/n) * np.sum(diff*diff)
    return loss


def mse_gradient(y_hat,y):
    n = y.shape[1]
    return (2/n) * (y_hat - y)


def cross_entropy_forward(y_hat,y):
    n = y.shape[1]
    shifted = y_hat - np.max(y_hat,axis=0,keepdims=True)
    exp_vals = np.exp(shifted)
    probs = exp_vals / (np.sum(exp_vals,axis=0,keepdims=True) + 1e-9)
    loss = -(1/n) * np.sum(y * np.log(probs + 1e-9))
    return loss,probs


def cross_entropy_gradient(y_hat,y):
    n = y.shape[1]
    return (1/n) * (y_hat - y)


class LossLayer:
    def __init__(self,loss_type):
        self.loss_type = loss_type
        self.y_true = None
        self.y_hat = None
        self.probs = None

    def forward_pass(self,y_hat,y):
        self.y_true = y
        self.y_hat = y_hat
        if self.loss_type == "mse":
            return mse_forward(y_hat,y)
        if self.loss_type == "cross_entropy":
            loss,self.probs = cross_entropy_forward(y_hat,y)
            return loss

    def backward_pass(self):
        if self.loss_type == "mse":
            return mse_gradient(self.y_hat,self.y_true)
        if self.loss_type == "cross_entropy":
            return cross_entropy_gradient(self.probs,self.y_true)
