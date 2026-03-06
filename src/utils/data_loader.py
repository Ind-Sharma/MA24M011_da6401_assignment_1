"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""
import numpy as np


def _load_via_urllib(dataset_name):
    """Download MNIST/Fashion-MNIST directly using urllib (no ML framework needed)."""
    import gzip
    import os
    import urllib.request

    if dataset_name == "mnist":
        base_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
        files = {
            "train_images": "train-images-idx3-ubyte.gz",
            "train_labels": "train-labels-idx1-ubyte.gz",
            "test_images":  "t10k-images-idx3-ubyte.gz",
            "test_labels":  "t10k-labels-idx1-ubyte.gz",
        }
    else:
        base_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
        files = {
            "train_images": "train-images-idx3-ubyte.gz",
            "train_labels": "train-labels-idx1-ubyte.gz",
            "test_images":  "t10k-images-idx3-ubyte.gz",
            "test_labels":  "t10k-labels-idx1-ubyte.gz",
        }
        base_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/fashion-mnist/"

    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "datasets", dataset_name)
    os.makedirs(cache_dir, exist_ok=True)

    def load_images(fname):
        path = os.path.join(cache_dir, fname)
        if not os.path.exists(path):
            urllib.request.urlretrieve(base_url + fname, path)
        with gzip.open(path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 28, 28)

    def load_labels(fname):
        path = os.path.join(cache_dir, fname)
        if not os.path.exists(path):
            urllib.request.urlretrieve(base_url + fname, path)
        with gzip.open(path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    X_train = load_images(files["train_images"])
    y_train = load_labels(files["train_labels"])
    X_test  = load_images(files["test_images"])
    y_test  = load_labels(files["test_labels"])
    return (X_train, y_train), (X_test, y_test)


def load_dataset(dataset_name):
    loaded = False
    # Try keras standalone first (lighter import, works with keras 2.x)
    if not loaded:
        try:
            import importlib
            keras_datasets = importlib.import_module("keras.datasets")
            if dataset_name == "mnist":
                (X_train,y_train),(X_test,y_test) = keras_datasets.mnist.load_data()
            else:
                (X_train,y_train),(X_test,y_test) = keras_datasets.fashion_mnist.load_data()
            loaded = True
        except Exception:
            pass
    # Fallback: tensorflow.keras
    if not loaded:
        try:
            import tensorflow as tf
            if dataset_name == "mnist":
                (X_train,y_train),(X_test,y_test) = tf.keras.datasets.mnist.load_data()
            else:
                (X_train,y_train),(X_test,y_test) = tf.keras.datasets.fashion_mnist.load_data()
            loaded = True
        except Exception:
            pass
    # Final fallback: raw urllib download (no ML framework needed)
    if not loaded:
        (X_train,y_train),(X_test,y_test) = _load_via_urllib(dataset_name)

    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_test  = X_test.reshape(X_test.shape[0],  -1) / 255.0
    return X_train, y_train, X_test, y_test


def one_hot_encode(y,num_classes=10):
    m = y.shape[0]
    Y = np.zeros((m,num_classes))
    for i in range(m):
        Y[i, y[i]] = 1
    return Y


