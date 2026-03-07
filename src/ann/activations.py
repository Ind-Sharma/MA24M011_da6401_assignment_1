import numpy as np

def sigmoid(z):
    val = 1/(1+np.exp(-z))
    return val

def sigmoid_grad(z):
    s = sigmoid(z)
    ans = s*(1-s)
    return ans

def relu(z):
    return np.maximum(0,z)

def relu_grad(z):
    ans = (z>0).astype(float)
    return ans

def softmax(z):
    z2 = z - np.max(z,axis=0,keepdims=True)
    e = np.exp(z2)
    return e/np.sum(e,axis=0,keepdims=True)
