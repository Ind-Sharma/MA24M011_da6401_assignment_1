"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""
import numpy as np


def mse_forward(y_hat,y):
    n = y.shape[1]
    difference = y_hat-y
    mean_squared_error = (1/n)*np.sum(difference*difference)
    return mean_squared_error


def mse_gradient(y_hat,y):
    n = y.shape[1]
    gradient = (2/n)*(y_hat-y)
    return gradient


def cross_entropy_forward(y_hat,y):
    n = y.shape[1]
    # numerically stable softmax: subtract max per column
    shifted = y_hat - np.max(y_hat, axis=0, keepdims=True)
    exp_pred = np.exp(shifted)
    probabilities = exp_pred / (np.sum(exp_pred, axis=0, keepdims=True) + 1e-9)
    # cross entropy loss
    loss = -(1/n)*np.sum(y*np.log(probabilities+1e-9))  # 1e-9 to avoid log(0)
    return loss, probabilities


def cross_entropy_gradient(y_hat,y):
    n = y.shape[1]
    gradient = (1/n)*(y_hat - y)
    return gradient


class LossLayer:
    def __init__(self, loss_type):
        self.loss_type = loss_type
        self.y_true = None
        self.y_hat = None
        self.probabilities = None

    def forward_pass(self,y_hat,y):
        self.y_true = y
        self.y_hat = y_hat

        if self.loss_type == "mse":
            return mse_forward(y_hat,y)

        if self.loss_type == "cross_entropy":
            loss, self.probabilities = cross_entropy_forward(y_hat,y)
            return loss

    def backward_pass(self):
        n = self.y_true.shape[1]

        if self.loss_type == "mse":
            return mse_gradient(self.y_hat,self.y_true)

        if self.loss_type == "cross_entropy":
            return cross_entropy_gradient(self.probabilities,self.y_true)