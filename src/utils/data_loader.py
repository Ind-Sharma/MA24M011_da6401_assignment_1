import numpy as np


def load_dataset(dataset_name):
    loaded = False

    # 1. Try sklearn with liac-arff parser (no pandas needed)
    if dataset_name == 'mnist':
        try:
            from sklearn.datasets import fetch_openml
            data = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
            X = data.data.astype(float)
            y = data.target.astype(int)
            X_train, X_test = X[:60000], X[60000:]
            y_train, y_test = y[:60000], y[60000:]
            loaded = True
        except Exception:
            pass

    # 2. tensorflow.keras
    if not loaded:
        import tensorflow as tf
        if dataset_name == 'mnist':
            (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        else:
            (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], -1).astype(float) / 255.0
    X_test  = X_test.reshape(X_test.shape[0],  -1).astype(float) / 255.0
    return X_train, y_train.astype(int), X_test, y_test.astype(int)
