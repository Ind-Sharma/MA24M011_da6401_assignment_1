import numpy as np

def sigmoid(z):
    val = 1  /  (1+np.exp(-z)) # standard sigmoid formula
    return val

def sigmoid_grad(z):
    s = sigmoid(z) # reuse sigmoid
    ans = s * (1-s) # derivaitve
    return ans

def relu(z):
    return np.maximum(0 ,z) # negaitve values become 0

def relu_grad(z):
    ans = (z >0).astype(float) # 1 where postive else 0
    return ans

def softmax(z):
    z2 = z - np.max(z,axis=0,keepdims=True) # shift for stabillity
    e = np.exp(z2)
    return e/np.sum(e,axis=0,keepdims=True) # normalise
