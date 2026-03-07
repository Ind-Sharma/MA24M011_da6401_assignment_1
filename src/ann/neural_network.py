import numpy as np
from .objective_functions import LossLayer
from .optimizers import SGD,Momentum,RMSprop
from .neural_layer import NNLayer,ActivationLayer


def _build_network(args):
    inp_dim = 784
    out_dim = 10
    h = getattr(args,'hidden_layers', getattr(args,'hidden_size',[128]))
    if isinstance(h, int):
        h = [h]
    weight_init = getattr(args,'weight_init','xavier')
    activation = getattr(args,'activation','relu')

    layers = []
    prev_size = inp_dim
    for curr_size in h:
        layers.append(NNLayer(curr_size,prev_size,init=weight_init))
        layers.append(ActivationLayer(activation))
        prev_size = curr_size
    layers.append(NNLayer(out_dim,prev_size,init=weight_init))
    return layers


def _get_optimizer(args):
    lr = getattr(args,'learning_rate',0.01)
    wd = getattr(args,'weight_decay',0.0)
    opt = getattr(args,'optimizer','sgd')
    if opt == 'sgd':
        return SGD(lr,wd)
    elif opt == 'momentum':
        return Momentum(lr,wd)
    elif opt == 'rmsprop':
        return RMSprop(lr,wd)
    return SGD(lr,wd)


class NeuralNetwork:
    def __init__(self,args):
        if not hasattr(args,'hidden_layers'):
            args.hidden_layers = getattr(args,'hidden_size',[128])

        self._activation = getattr(args,'activation','relu')
        self.layers = _build_network(args)
        self.param_layers = [l for l in self.layers if hasattr(l,'grad_W') and hasattr(l,'W')]
        self.loss_fn = LossLayer(getattr(args,'loss','cross_entropy'))
        self.optimizer = _get_optimizer(args)

    def forward(self,X):
        a = X.T
        for layer in self.layers:
            a = layer.forward_pass(a)
        return a.T

    def backward(self,y,y_hat):
        grad_W_list = []
        grad_b_list = []

        if y.ndim == 1 or (y.ndim == 2 and min(y.shape) == 1):
            n_classes = y_hat.shape[1] if y_hat.ndim == 2 else 10
            y_flat = y.astype(int).flatten()
            y_oh = np.zeros((len(y_flat),n_classes))
            y_oh[np.arange(len(y_flat)),y_flat] = 1.0
            y = y_oh

        loss = self.loss_fn.forward_pass(y_hat.T,y.T)
        grad_from_next = self.loss_fn.backward_pass()

        for current_layer in reversed(self.layers):
            grad_from_next = current_layer.backward_pass(grad_from_next)
            if hasattr(current_layer,'grad_W') and hasattr(current_layer,'W'):
                grad_W_list.append(current_layer.grad_W.T)
                grad_b_list.append(current_layer.grad_b)

        self.grad_W = grad_W_list
        self.grad_b = grad_b_list
        return self.grad_W,self.grad_b

    def update_weights(self):
        self.optimizer.update(self.param_layers)

    def train(self,X_train,y_train,epochs=1,batch_size=32):
        m = len(y_train)
        Y = np.zeros((m,10))
        Y[np.arange(m),y_train.astype(int)] = 1
        Y = Y.T

        N = X_train.shape[0]
        avg_loss = 0.0
        for ep in range(epochs):
            perm = np.random.permutation(N)
            X_shuf = X_train[perm]
            Y_shuf = Y[:,perm]
            total_loss = 0.0
            batches = range(0,N,batch_size)
            for start in batches:
                X_b = X_shuf[start:start+batch_size]
                Y_b = Y_shuf[:,start:start+batch_size]
                y_hat = self.forward(X_b)
                total_loss += self.loss_fn.forward_pass(y_hat.T,Y_b)
                self.backward(Y_b.T,y_hat)
                self.update_weights()
            avg_loss = total_loss / len(batches)
        return avg_loss

    def evaluate(self,X,y):
        from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score as f1_fn

        y_hat = self.forward(np.array(X,dtype=float))
        preds = np.argmax(y_hat,axis=1)
        y = np.array(y,dtype=int).flatten()

        return {
            "accuracy": float(accuracy_score(y,preds)),
            "precision": float(precision_score(y,preds,average='macro',zero_division=0)),
            "recall": float(recall_score(y,preds,average='macro',zero_division=0)),
            "f1": float(f1_fn(y,preds,average='macro',zero_division=0)),
        }

    def get_weights(self):
        d = {}
        for i,layer in enumerate(self.param_layers):
            d["W"+str(i)] = layer.W.copy()
            d["b"+str(i)] = layer.b.copy()
        d['_activation'] = self._activation
        return d

    def set_weights(self,weights):
        if isinstance(weights,dict) and '_activation' in weights:
            self._activation = weights['_activation']
            for layer in self.layers:
                if hasattr(layer,'activation'):
                    layer.activation = self._activation

        W_list = {}
        b_list = {}
        for k in weights:
            if k.startswith('W') and k[1:].isdigit():
                W_list[int(k[1:])] = np.array(weights[k],dtype=float)
            elif k.startswith('b') and k[1:].isdigit():
                b_list[int(k[1:])] = np.array(weights[k],dtype=float)
        W_sorted = [W_list[k] for k in sorted(W_list)]
        b_sorted = [b_list[k] for k in sorted(b_list)]

        if len(W_sorted) == len(self.param_layers):
            for i in range(len(self.param_layers)):
                W = W_sorted[i]
                if W.shape != self.param_layers[i].W.shape and W.T.shape == self.param_layers[i].W.shape:
                    W = W.T
                self.param_layers[i].W = W
                self.param_layers[i].b = b_sorted[i].flatten().reshape(-1,1)
        else:
            new_layers = []
            for i in range(len(W_sorted)):
                W = W_sorted[i]
                b = b_sorted[i]
                layer = NNLayer(W.shape[0],W.shape[1])
                layer.W = W.copy()
                layer.b = b.flatten().reshape(-1,1)
                new_layers.append(layer)
                if i < len(W_sorted) - 1:
                    new_layers.append(ActivationLayer(self._activation))
            self.layers = new_layers
            self.param_layers = [l for l in self.layers if hasattr(l,'W')]
