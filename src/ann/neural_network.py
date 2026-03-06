"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np
from .objective_functions import LossLayer
from .optimizers import SGD, Momentum, RMSprop
from .neural_layer import NNLayer


def _get_optimizer(args):
    lr = getattr(args, 'learning_rate', 0.01)
    wd = getattr(args, 'weight_decay', 0.0)
    opt = getattr(args, 'optimizer', 'sgd')

    if opt == 'sgd':
        return SGD(lr, wd)
    elif opt == 'momentum':
        return Momentum(lr, wd)
    elif opt == 'rmsprop':
        return RMSprop(lr, wd)
    else:
        return SGD(lr, wd)


class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    Accepts CLI args and builds network.
    Forward returns logits (no softmax) - required for MSE.
    Backward returns arrays of arrays, index 0 = last layer gradient as instructed in class..
    """

    def __init__(self, args):
        # Support both 'hidden_size' (train.py) and 'hidden_layers' (inference.py / grader)
        if not hasattr(args, 'hidden_layers'):
            args.hidden_layers = getattr(args, 'hidden_size', [128])
        if not hasattr(args, 'hidden_size'):
            args.hidden_size = args.hidden_layers

        try:
            from ..utils.data_loader import build_network
        except ImportError:
            try:
                from src.utils.data_loader import build_network
            except ImportError:
                from utils.data_loader import build_network
        self.layers = build_network(args)
        self.param_layers = [layer for layer in self.layers if type(layer) == NNLayer]

        self.loss_fn = LossLayer(getattr(args, 'loss', 'cross_entropy'))
        self.optimizer = _get_optimizer(args)

    def forward(self, X):
        """
        Forward propagation through all layers.
        Returns logits (no softmax applied)
        X is shape (b, D_in) and output is shape (b, D_out).
        b is batch size, D_in is input dimension, D_out is output dimension.
        """
        a = X.T
        for layer in self.layers:
            a = layer.forward_pass(a)
        return a.T

    def backward(self,y,y_hat):
        """
        Backward propagation to compute gradients.
        Returns two numpy arrays: grad_Ws, grad_bs.
        - `grad_Ws[0]` is gradient for the last (output) layer weights,
          `grad_bs[0]` is gradient for the last layer biases, and so on.
        y can be one-hot (batch, classes) or integer labels (batch,).
        """
        grad_W_list = []
        grad_b_list = []

        # Convert integer labels to one-hot if needed
        if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):
            n_classes = y_hat.shape[1] if y_hat.ndim == 2 else 10
            y_oh = np.zeros((len(y), n_classes))
            y_oh[np.arange(len(y)), y.astype(int).flatten()] = 1.0
            y = y_oh

        # Loss expects (classes, batch); forward returns (batch, classes)
        loss = self.loss_fn.forward_pass(y_hat.T, y.T)
        grad_from_next = self.loss_fn.backward_pass()

        num_layers = len(self.layers)
        layer_index = num_layers-1
        while layer_index >= 0:
            current_layer = self.layers[layer_index]
            grad_from_next = current_layer.backward_pass(grad_from_next)
            if type(current_layer) == NNLayer:
                grad_W_list.append(current_layer.grad_W)
                grad_b_list.append(current_layer.grad_b)
            layer_index = layer_index-1

        self.grad_W = grad_W_list
        self.grad_b = grad_b_list

        return self.grad_W, self.grad_b

    def update_weights(self):
        self.optimizer.update(self.param_layers)

    def train(self,X_train,y_train,epochs=1,batch_size=32):
        try:
            from ..utils.data_loader import one_hot_encode
        except ImportError:
            try:
                from src.utils.data_loader import one_hot_encode
            except ImportError:
                from utils.data_loader import one_hot_encode

        Y = one_hot_encode(y_train).T
        N = X_train.shape[0]
        for ep in range(epochs):
            perm = np.random.permutation(N)
            X_shuf = X_train[perm]
            Y_shuf = Y[:,perm]
            total_loss = 0.0
            count = 0
            for start in range(0,N,batch_size):
                end = start+batch_size
                if end > N:
                    end = N
                X_b = X_shuf[start:end]
                Y_b = Y_shuf[:,start:end]
                y_hat = self.forward(X_b)
                loss = self.loss_fn.forward_pass(y_hat.T, Y_b)
                self.backward(Y_b.T, y_hat)
                total_loss = total_loss+loss
                count = count+1
                self.update_weights()
            avg_loss = total_loss/count
        return avg_loss

    def evaluate(self,X,y):
        from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score as f1_score_fn,confusion_matrix as confusion_matrix_fn

        y_hat = self.forward(X)
        y_hat_labels = np.argmax(y_hat,axis=1)

        accuracy = accuracy_score(y,y_hat_labels)
        precision = precision_score(y, y_hat_labels,average='macro',zero_division=0)
        recall = recall_score(y, y_hat_labels,average='macro',zero_division=0)
        f1 = f1_score_fn(y, y_hat_labels,average='macro',zero_division=0)
        cm = confusion_matrix_fn(y,y_hat_labels)

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "confusion_matrix": cm
        }

    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.param_layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weights):
        if isinstance(weights, list):
            # List of (W, b) tuples or flat list of arrays
            if len(weights) > 0 and isinstance(weights[0], tuple):
                for i, (W, b) in enumerate(weights):
                    if i < len(self.param_layers):
                        self.param_layers[i].W = W.copy()
                        self.param_layers[i].b = b.copy()
            else:
                # flat list: [W0, b0, W1, b1, ...]
                for i, layer in enumerate(self.param_layers):
                    if 2*i < len(weights):
                        layer.W = weights[2*i].copy()
                    if 2*i+1 < len(weights):
                        layer.b = weights[2*i+1].copy()
        else:
            # Dict format - support both 0-indexed (W0) and 1-indexed (W1)
            for i, layer in enumerate(self.param_layers):
                for w_key in [f"W{i}", f"W{i+1}"]:
                    if w_key in weights:
                        layer.W = weights[w_key].copy()
                        break
                for b_key in [f"b{i}", f"b{i+1}"]:
                    if b_key in weights:
                        layer.b = weights[b_key].copy()
                        break
