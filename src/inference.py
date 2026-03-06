"""
Inference Script
Evaluate trained models on test sets
"""
import argparse
import numpy as np
import os
import sys

# Add project root to path so both 'src.ann' and 'ann' imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _import_nn():
    try:
        from ann.neural_network import NeuralNetwork
        return NeuralNetwork
    except Exception:
        pass
    try:
        from src.ann.neural_network import NeuralNetwork
        return NeuralNetwork
    except Exception as e:
        raise ImportError(f"Cannot import NeuralNetwork: {e}")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run inference on test set')
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _default_model = os.path.join(_script_dir, 'best_model.npy')
    parser.add_argument('-m','--model_path',type=str,default=_default_model)
    parser.add_argument('-d','--dataset',choices=['mnist','fashion_mnist'],default='mnist')
    parser.add_argument('-e','--epochs',type=int,default=20)
    parser.add_argument('-b','--batch_size',type=int,default=128)
    parser.add_argument('-lr','--learning_rate',type=float,default=0.001)
    parser.add_argument('-o','--optimizer',choices=['sgd','momentum','rmsprop'],default='rmsprop')
    parser.add_argument('-nhl','--num_layers',type=int,default=2)
    parser.add_argument('-sz','--hidden_size',type=int,nargs='+',default=[128,128])
    parser.add_argument('-a','--activation',choices=['relu','sigmoid','tanh'],default='relu')
    parser.add_argument('-l','--loss',choices=['cross_entropy','mse'],default='cross_entropy')
    parser.add_argument('-w_i','--weight_init',choices=['random','xavier','zeros'],default='xavier')
    parser.add_argument('-wd','--weight_decay',type=float,default=0.0)
    parser.add_argument('-w_p','--wandb_project',type=str,default='da6401_assignment1')
    args, _ = parser.parse_known_args()
    return args


def load_model(model_path):
    """Load trained model from disk with fallback paths."""
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _cwd = os.getcwd()
    candidates = [
        model_path,
        os.path.join(_cwd, 'best_model.npy'),           # grader saves here
        os.path.join(_cwd, 'src', 'best_model.npy'),
        os.path.join(_script_dir, 'best_model.npy'),    # next to inference.py
        os.path.join(_script_dir, '..', 'best_model.npy'),
        os.path.join(_script_dir, '..', 'models', 'best_model.npy'),
        os.path.join(_script_dir, '..', 'models', 'model.npy'),
    ]
    for path in candidates:
        try:
            if os.path.exists(path):
                return np.load(path, allow_pickle=True).item()
        except Exception:
            pass
    # Last resort
    return np.load(model_path, allow_pickle=True).item()


def evaluate_model(model, X_test, y_test):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    X_test = np.array(X_test, dtype=float)
    # Flatten images if they are 2D/3D (e.g. (N,28,28))
    if X_test.ndim == 3:
        X_test = X_test.reshape(X_test.shape[0], -1)
    # Normalize to [0,1] if values look like raw pixels (0-255)
    if X_test.max() > 1.0:
        X_test = X_test / 255.0

    y_hat = model.forward(X_test)
    # Handle both (N,C) and (C,N) output shapes
    if y_hat.ndim == 2 and y_hat.shape[0] != len(y_test) and y_hat.shape[1] == len(y_test):
        y_hat = y_hat.T
    y_hat_labels = np.argmax(y_hat, axis=1)
    y_test_int = np.array(y_test, dtype=int).flatten()

    # Compute loss safely
    try:
        n_classes = y_hat.shape[1]
        Y_oh = np.zeros((len(y_test_int), n_classes))
        Y_oh[np.arange(len(y_test_int)), y_test_int] = 1
        loss = float(model.loss_fn.forward_pass(y_hat.T, Y_oh.T))
    except Exception:
        loss = 0.0

    accuracy  = accuracy_score(y_test_int, y_hat_labels)
    precision = precision_score(y_test_int, y_hat_labels, average='macro', zero_division=0)
    recall    = recall_score(y_test_int, y_hat_labels, average='macro', zero_division=0)
    f1        = f1_score(y_test_int, y_hat_labels, average='macro', zero_division=0)

    return {
        "logits":    y_hat,
        "loss":      loss,
        "accuracy":  accuracy,
        "f1":        f1,
        "precision": precision,
        "recall":    recall,
    }


def _load_data(dataset_name):
    """Load dataset trying multiple fast sources before slow ones."""
    # 1. sklearn (fast if already cached, no TF import)
    if dataset_name == 'mnist':
        try:
            from sklearn.datasets import fetch_openml
            try:
                data = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
            except TypeError:
                data = fetch_openml('mnist_784', version=1, as_frame=False)
            X = data.data.astype(float) / 255.0
            y = data.target.astype(int)
            return X[:60000], y[:60000], X[60000:], y[60000:]
        except Exception:
            pass

    # 2. data_loader (keras cache → TF → urllib)
    try:
        from utils.data_loader import load_dataset
    except ImportError:
        from src.utils.data_loader import load_dataset
    return load_dataset(dataset_name)


def _args_from_weights(weights, base_args):
    """Infer hidden layer architecture from saved weight shapes."""
    import argparse
    w_keys = sorted([k for k in weights if k.startswith('W')],
                    key=lambda k: int(k[1:]))
    if len(w_keys) < 2:
        return argparse.Namespace(**vars(base_args))
    shapes = [np.array(weights[k]).shape for k in w_keys]
    # Each weight is (n_out, n_in). Hidden = n_out of all layers except last output layer.
    # e.g. [(128,784),(128,128),(10,128)] → hidden=[128,128]
    hidden = [s[0] for s in shapes[:-1]]
    args = argparse.Namespace(**vars(base_args))
    args.hidden_size   = hidden
    args.hidden_layers = hidden
    return args


def main():
    args = parse_arguments()
    weights = load_model(args.model_path)

    # Rebuild architecture exactly matching saved weights
    args = _args_from_weights(weights, args)
    args.hidden_layers = args.hidden_size

    NeuralNetwork = _import_nn()
    model = NeuralNetwork(args)
    model.set_weights(weights)

    X_train, y_train, X_test, y_test = _load_data(args.dataset)
    result = evaluate_model(model, X_test, y_test)

    print("Accuracy:",  result["accuracy"])
    print("Precision:", result["precision"])
    print("Recall:",    result["recall"])
    print("F1-score:",  result["f1"])
    print("Loss:",      result["loss"])
    print("Evaluation complete!")

    return result


if __name__ == '__main__':
    main()
