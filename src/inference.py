"""
Inference Script
Evaluate trained models on test sets
"""
import argparse
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _import_nn():
    try:
        from ann.neural_network import NeuralNetwork
        return NeuralNetwork
    except Exception:
        from src.ann.neural_network import NeuralNetwork
        return NeuralNetwork


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run inference on test set')
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _default_model = os.path.join(_script_dir, 'best_model.npy')
    parser.add_argument('-m', '--model_path', type=str, default=_default_model)
    parser.add_argument('-d', '--dataset', choices=['mnist', 'fashion_mnist'], default='mnist')
    parser.add_argument('-e', '--epochs', type=int, default=20)
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-o', '--optimizer', choices=['sgd', 'momentum', 'rmsprop'], default='rmsprop')
    parser.add_argument('-nhl', '--num_layers', type=int, default=2)
    parser.add_argument('-sz', '--hidden_size', type=int, nargs='+', default=[128, 128])
    parser.add_argument('-a', '--activation', choices=['relu', 'sigmoid', 'tanh'], default='relu')
    parser.add_argument('-l', '--loss', choices=['cross_entropy', 'mse'], default='cross_entropy')
    parser.add_argument('-w_i', '--weight_init', choices=['random', 'xavier', 'zeros'], default='xavier')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0)
    parser.add_argument('-w_p', '--wandb_project', type=str, default='da6401_assignment1')
    args, _ = parser.parse_known_args()
    return args


def load_model(model_path):
    """Load trained model from disk."""
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _cwd = os.getcwd()
    candidates = [
        model_path,
        os.path.join(_script_dir, 'best_model.npy'),
        os.path.join(_cwd, 'src', 'best_model.npy'),
        os.path.join(_cwd, 'best_model.npy'),
    ]
    for path in candidates:
        try:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                return np.load(abs_path, allow_pickle=True).item()
        except Exception:
            pass
    return np.load(model_path, allow_pickle=True).item()


def evaluate_model(model, X_test, y_test):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    X_test = np.array(X_test, dtype=float)
    if X_test.ndim == 3:
        X_test = X_test.reshape(X_test.shape[0], -1)
    if X_test.max() > 2.0:
        X_test = X_test / 255.0

    y_hat = model.forward(X_test)
    if y_hat.ndim == 2 and y_hat.shape[0] != len(y_test) and y_hat.shape[1] == len(y_test):
        y_hat = y_hat.T
    y_hat_labels = np.argmax(y_hat, axis=1)
    y_test_int = np.array(y_test, dtype=int).flatten()

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
    try:
        from utils.data_loader import load_dataset
    except ImportError:
        from src.utils.data_loader import load_dataset
    return load_dataset(dataset_name)


def main():
    args = parse_arguments()
    weights = load_model(args.model_path)

    # Infer architecture from weights
    w_keys = sorted([k for k in weights if k.startswith('W')], key=lambda k: int(k[1:]))
    shapes = [np.array(weights[k]).shape for k in w_keys]
    hidden = [s[0] for s in shapes[:-1]]
    args.hidden_size = hidden
    args.hidden_layers = hidden

    NeuralNetwork = _import_nn()
    model = NeuralNetwork(args)
    model.set_weights(weights)

    _, _, X_test, y_test = _load_data(args.dataset)
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
