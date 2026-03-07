import numpy as np


class LossLayer:
    def __init__(self,loss_type):
        self.loss_type = loss_type
        self.y_true = None
        self.probs = None

    def forward_pass(self,y_hat,y):
        self.y_true = y
        n = y.shape[1]
        if self.loss_type == "mse":
            self.probs = y_hat
            return (1/n) * np.sum((y_hat - y)**2)
        shifted = y_hat - np.max(y_hat,axis=0,keepdims=True)
        exp_vals = np.exp(shifted)
        self.probs = exp_vals / (np.sum(exp_vals,axis=0,keepdims=True) + 1e-9)
        return -(1/n) * np.sum(y * np.log(self.probs + 1e-9))

    def backward_pass(self):
        n = self.y_true.shape[1]
        if self.loss_type == "mse":
            return (2/n) * (self.probs - self.y_true)
        return (1/n) * (self.probs - self.y_true)
