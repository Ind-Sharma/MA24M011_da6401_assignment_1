import numpy as np

class LossLayer:
    def __init__(self,loss_type):
        self.loss_type = loss_type # mse or cross_entopy
        self.y_true = None
        self.probs  = None

    def forward_pass(self,y_hat,y):
        self.y_true = y # store for backword
        n = y.shape[ 1] # numbr of samples
        if self.loss_type=="mse":
            self.probs = y_hat
            loss = (1/n) *np.sum((y_hat-y)**2) # mean sqaured error
            return loss
        z = y_hat - np.max(y_hat ,axis=0,keepdims=True) # shift vallues
        e = np.exp(z)
        self.probs = e/(np.sum(e,axis=0,keepdims=True)+1e-9) # sotfmax
        loss = -(1/n)*np.sum(y*np.log(self.probs+1e-9)) # cross entopy
        return loss

    def backward_pass(self):
        n = self.y_true.shape[1] # btach size
        if self.loss_type=="mse":
            return (2/n)*(self.probs-self.y_true) # mse graident
        return (1/n)*(self.probs-self.y_true) # cross entopy graident
