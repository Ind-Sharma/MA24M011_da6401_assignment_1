import numpy as np
from .objective_functions import LossLayer
from .optimizers import SGD,Momentum,RMSprop
from .neural_layer import NNLayer,ActivationLayer

class NeuralNetwork:
    def __init__(self,args):
        if not hasattr(args,'hidden_layers'):
            args.hidden_layers = getattr(args,'hidden_size',[128]) # fallback
        self._activation = getattr(args,'activation','relu')
        wi = getattr(args,'weight_init','xavier')
        h = args.hidden_layers
        if isinstance(h,int):
            h = [h] # make it a list
        layers = []
        prev = 784 # input size for mnist
        for sz in h:
            layers.append(NNLayer(sz,prev,init=wi)) # hidden layer
            layers.append(ActivationLayer(self._activation)) # activation
            prev = sz
        layers.append(NNLayer(10,prev,init=wi)) # output layer 10 classes
        self.layers = layers
        self.param_layers = [l for l in layers if hasattr(l,'W')] # only weight layers
        self.loss_fn = LossLayer(getattr(args,'loss','cross_entropy'))
        lr = getattr(args,'learning_rate',0.01)
        wd = getattr(args,'weight_decay',0.0)
        opt = getattr(args,'optimizer','sgd')
        if opt=='momentum':
            self.optimizer = Momentum(lr,wd)
        elif opt=='rmsprop':
            self.optimizer = RMSprop(lr,wd)
        else:
            self.optimizer = SGD(lr,wd) # default

    def forward(self,X):
        a = X.T # transpose so shape is (features, batch)
        for layer in self.layers:
            a = layer.forward_pass(a)
        return a.T # back to (batch, classes)

    def backward(self,y,y_hat):
        if y.ndim==1 or (y.ndim==2 and min(y.shape)==1):
            yf = y.astype(int).flatten()
            yoh = np.zeros((len(yf),10))
            yoh[np.arange(len(yf)),yf] = 1.0 # one hot encode
            y = yoh
        self.loss_fn.forward_pass(y_hat.T,y.T)
        g = self.loss_fn.backward_pass() # start gradient from loss
        gW = []
        gb = []
        for layer in reversed(self.layers): # go backwards
            g = layer.backward_pass(g)
            if hasattr(layer,'grad_W'):
                gW.append(layer.grad_W.T)
                gb.append(layer.grad_b)
        return gW,gb

    def train(self,X_train,y_train,epochs=1,batch_size=32):
        m = len(y_train)
        Y = np.zeros((m,10))
        Y[np.arange(m),y_train.astype(int)] = 1 # one hot labels
        Y = Y.T
        N = X_train.shape[0] # total samples
        avg_loss = 0.0
        for _ in range(epochs):
            idx = np.random.permutation(N) # shuffle
            Xs = X_train[idx]
            Ys = Y[:,idx]
            total = 0.0
            steps = range(0,N,batch_size)
            for s in steps:
                xb = Xs[s:s+batch_size] # get batch
                yb = Ys[:,s:s+batch_size]
                yhat = self.forward(xb)
                total += self.loss_fn.forward_pass(yhat.T,yb) # compute loss
                self.backward(yb.T,yhat)
                self.optimizer.update(self.param_layers) # update weights
            avg_loss = total/len(steps)
        return avg_loss

    def evaluate(self,X,y):
        from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score as f1_fn
        yhat = self.forward(np.array(X,dtype=float))
        preds = np.argmax(yhat,axis=1) # pick highest score
        y = np.array(y,dtype=int).flatten()
        return {
            "accuracy":float(accuracy_score(y,preds)),
            "precision":float(precision_score(y,preds,average='macro',zero_division=0)),
            "recall":float(recall_score(y,preds,average='macro',zero_division=0)),
            "f1":float(f1_fn(y,preds,average='macro',zero_division=0)),
        }

    def get_weights(self):
        d = {}
        for i,layer in enumerate(self.param_layers):
            d["W"+str(i)] = layer.W.copy() # save weight matrix
            d["b"+str(i)] = layer.b.copy() # save bias
        d['_activation'] = self._activation # save activation type
        return d

    def set_weights(self,weights):
        if '_activation' in weights:
            self._activation = weights['_activation']
            for layer in self.layers:
                if hasattr(layer,'activation'):
                    layer.activation = self._activation # update activation
        Wdict = {int(k[1:]):np.array(weights[k],dtype=float) for k in weights if k.startswith('W') and k[1:].isdigit()}
        bdict = {int(k[1:]):np.array(weights[k],dtype=float) for k in weights if k.startswith('b') and k[1:].isdigit()}
        Wlist = [Wdict[k] for k in sorted(Wdict)]
        blist = [bdict[k] for k in sorted(bdict)]
        if len(Wlist)==len(self.param_layers):
            for i,layer in enumerate(self.param_layers):
                W = Wlist[i]
                if W.shape!=layer.W.shape and W.T.shape==layer.W.shape:
                    W = W.T # fix orientation if needed
                layer.W = W
                layer.b = blist[i].flatten().reshape(-1,1)
        else:
            nl = []
            for i,W in enumerate(Wlist):
                layer = NNLayer(W.shape[0],W.shape[1])
                layer.W = W.copy()
                layer.b = blist[i].flatten().reshape(-1,1)
                nl.append(layer)
                if i<len(Wlist)-1:
                    nl.append(ActivationLayer(self._activation))
            self.layers = nl
            self.param_layers = [l for l in self.layers if hasattr(l,'W')]
