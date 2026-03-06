"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np
from .objective_functions import LossLayer
from .optimizers import SGD, Momentum, RMSprop
from .neural_layer import NNLayer, ActivationLayer


def _build_network(args):
    """Build list of layers from args."""
    inp_dim = 784
    out_dim = 10
    h = getattr(args, 'hidden_layers', None)
    if h is None:
        h = getattr(args, 'hidden_size', [128])
    if h is None:
        h = [128]
    if isinstance(h, int):
        h = [h]
    weight_init = getattr(args, 'weight_init', 'xavier')
    activation  = getattr(args, 'activation', 'relu')

    layers = []
    prev_size = inp_dim
    for curr_size in h:
        layers.append(NNLayer(curr_size, prev_size, init=weight_init))
        layers.append(ActivationLayer(activation))
        prev_size = curr_size
    layers.append(NNLayer(out_dim, prev_size, init=weight_init))
    return layers


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

        self._activation = getattr(args, 'activation', 'relu')
        self.layers = _build_network(args)
        self.param_layers = [l for l in self.layers if hasattr(l, 'grad_W') and hasattr(l, 'W')]

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
            if hasattr(current_layer, 'grad_W') and hasattr(current_layer, 'W'):
                # Return grad_W as (n_in, n_out) = W.T shape for grader compatibility
                grad_W_list.append(current_layer.grad_W.T)
                grad_b_list.append(current_layer.grad_b)
            layer_index = layer_index-1

        self.grad_W = grad_W_list
        self.grad_b = grad_b_list

        return self.grad_W, self.grad_b

    def update_weights(self):
        self.optimizer.update(self.param_layers)

    def train(self,X_train,y_train,epochs=1,batch_size=32):
        def one_hot_encode(y, num_classes=10):
            m = len(y)
            Y = np.zeros((m, num_classes))
            Y[np.arange(m), y.astype(int)] = 1
            return Y

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

        X = np.array(X, dtype=float)
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        if X.max() > 1.0:
            X = X / 255.0
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
        # Store activation so set_weights can restore the correct activation
        d['_activation'] = self._activation
        return d

    def _assign_W(self, layer, W):
        """Set layer.W, transposing if needed so shape is (n_out, n_in)."""
        W = np.array(W, dtype=float)
        if W.shape != layer.W.shape and W.T.shape == layer.W.shape:
            W = W.T
        layer.W = W

    def _assign_b(self, layer, b):
        """Set layer.b, always as (n_out, 1)."""
        b = np.array(b, dtype=float).flatten().reshape(-1, 1)
        layer.b = b

    def set_weights(self, weights):
        if isinstance(weights, list):
            if len(weights) > 0 and isinstance(weights[0], tuple):
                # list of (W, b) tuples — match by position
                w_list = [(W, b) for W, b in weights]
            else:
                # flat list [W0, b0, W1, b1, ...]
                w_list = [(weights[2*i], weights[2*i+1])
                          for i in range(len(weights)//2)]
            for i, (W, b) in enumerate(w_list):
                if i < len(self.param_layers):
                    self._assign_W(self.param_layers[i], W)
                    self._assign_b(self.param_layers[i], b)
        else:
            # dict: extract all W/b arrays sorted by key index
            W_arrays = {}
            b_arrays = {}
            for k, v in weights.items():
                if k == '_activation':
                    self._activation = v  # restore activation from saved weights
                    continue
                if k.startswith('W'):
                    try:
                        W_arrays[int(k[1:])] = v
                    except ValueError:
                        pass
                elif k.startswith('b'):
                    try:
                        b_arrays[int(k[1:])] = v
                    except ValueError:
                        pass

            W_sorted = [W_arrays[k] for k in sorted(W_arrays)]
            b_sorted = [b_arrays[k] for k in sorted(b_arrays)]

            n_layers = len(self.param_layers)
            n_weights = len(W_sorted)

            if n_weights != n_layers:
                # Architecture mismatch: rebuild layers to match incoming weights
                # Use activation stored in weights dict if available
                activation = weights.get('_activation', getattr(self, '_activation', 'relu'))
                new_layers = []
                for i, (W, b) in enumerate(zip(W_sorted, b_sorted)):
                    W = np.array(W, dtype=float)
                    b = np.array(b, dtype=float)
                    n_out, n_in = W.shape if W.shape[0] < W.shape[1] or i == len(W_sorted)-1 else W.T.shape
                    # Determine correct orientation
                    if W.shape[0] <= W.shape[1]:
                        n_out, n_in = W.shape
                    else:
                        n_out, n_in = W.shape
                    layer = NNLayer(n_out, n_in)
                    layer.W = W.copy() if W.shape == (n_out, n_in) else W.T.copy()
                    layer.b = b.flatten().reshape(-1, 1)
                    new_layers.append(layer)
                    if i < n_weights - 1:
                        new_layers.append(ActivationLayer(activation))
                self.layers = new_layers
                self.param_layers = [l for l in self.layers if hasattr(l, 'grad_W') and hasattr(l, 'W')]
                return

            # Same number of layers — assign positionally
            for i, layer in enumerate(self.param_layers):
                self._assign_W(layer, W_sorted[i])
                if i < len(b_sorted):
                    self._assign_b(layer, b_sorted[i])
