import numpy as np

class LossLayer:
    def __init__(self,loss_type):
        self.loss_type = loss_type # mse or cross_entropy
        self.y_true = None
        self.probs = None

    def forward_pass(self,y_hat,y):
        self.y_true = y # store for backward
        n = y.shape[1] # number of samples
        if self.loss_type=="mse":
            self.probs = y_hat
            loss = (1/n)*np.sum((y_hat-y)**2) # mean squared error
            return loss
        z = y_hat - np.max(y_hat,axis=0,keepdims=True) # shift values
        e = np.exp(z)
        self.probs = e/(np.sum(e,axis=0,keepdims=True)+1e-9) # softmax
        loss = -(1/n)*np.sum(y*np.log(self.probs+1e-9)) # cross entropy
        return loss

    def backward_pass(self):
        n = self.y_true.shape[1] # batch size
        if self.loss_type=="mse":
            return (2/n)*(self.probs-self.y_true) # mse gradient
        return (1/n)*(self.probs-self.y_true) # cross entropy gradient
