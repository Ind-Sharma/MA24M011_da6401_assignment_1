"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""
import numpy as np
from .activations import sigmoid,sigmoid_grad,tanh,tanh_grad,relu,relu_grad,softmax,softmax_grad

def random_weight_init(m,n):
    W_list = np.empty((m,n))
    for i in range(m):
        for j in range(n):
            W_list[i,j] = np.random.randn()
    return W_list

def xavier_weight_init(m,n):
    W_list = np.empty((m,n))
    for i in range(m):
        for j in range(n):
            W_list[i,j] = np.random.randn()*(np.sqrt(1.0/n))
    return W_list

def zeros_weight_init(m,n):
    return np.zeros((m,n))

class NNLayer:
    def __init__(self,m,n,init="xavier"):
        num_neurons_current_layer=m
        num_neurons_prev_layer=n

        # initialize bias to zero
        self.b=np.zeros((num_neurons_current_layer,1))

        # initialize weights
        if init=="random":
            self.W=random_weight_init(num_neurons_current_layer,num_neurons_prev_layer)
        elif init=="zeros":
            self.W=zeros_weight_init(num_neurons_current_layer,num_neurons_prev_layer)
        else:
            self.W=xavier_weight_init(num_neurons_current_layer,num_neurons_prev_layer)

        self.prev_input=None
        self.grad_W=None
        self.grad_b=None
    
    def forward_pass(self,input):   
        self.prev_input=input
        linear_output=np.dot(self.W,input)          # Z = W ·A^(l-1)
        output=linear_output+self.b                 # Z = Z + b
        return output

    def backward_pass(self,dZ):
        batch_size = dZ.shape[1]  # dZ is (m,batch_size)

        # grad_W shape: (n_out, n_in) — same shape as W, correct for optimizer
        dZ_times_prev_input=np.dot(dZ,self.prev_input.T)
        self.grad_W=(1/batch_size)*dZ_times_prev_input

        # gradient of loss w.r.t. bias
        num_neurons = dZ.shape[0]
        sum_over_batch = np.zeros((num_neurons,1))
        for i in range(num_neurons):
            for j in range(batch_size):
                sum_over_batch[i,0] = sum_over_batch[i,0]+dZ[i,j]
        self.grad_b = (1/batch_size)*sum_over_batch

        # gradient to pass to previous layer
        grad_to_prev_layer=np.dot(self.W.T,dZ)

        return grad_to_prev_layer

class ActivationLayer:
    """we will use activation functions from activations.py for forward/backward."""
    def __init__(self, activation):
        self.activation = activation
        self.input = None

    def forward_pass(self,x):
        self.input = x
        if self.activation == "sigmoid":
            return sigmoid(x)
        if self.activation == "tanh":
            return tanh(x)
        if self.activation == "relu":
            return relu(x)
        if self.activation == "softmax":
            return softmax(x)

    def backward_pass(self,grad_from_next):
        if self.activation == "sigmoid":
            return grad_from_next*sigmoid_grad(self.input)
        if self.activation == "tanh":
            return grad_from_next*tanh_grad(self.input)
        if self.activation == "relu":
            return grad_from_next*relu_grad(self.input)
        if self.activation == "softmax":
            return grad_from_next*softmax_grad(self.input)
