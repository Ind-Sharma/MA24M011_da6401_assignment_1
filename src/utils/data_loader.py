import numpy as np
import gzip
import os


def load_dataset(dataset_name):
    folder = 'mnist' if dataset_name == 'mnist' else 'fashion-mnist'
    cache_dir = os.path.join(os.path.expanduser('~'), '.keras', 'datasets', folder)
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images':  't10k-images-idx3-ubyte.gz',
        'test_labels':  't10k-labels-idx1-ubyte.gz',
    }

    if all(os.path.exists(os.path.join(cache_dir, f)) for f in files.values()):
        def load_images(fname):
            with gzip.open(os.path.join(cache_dir, fname), 'rb') as f:
                return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)

        def load_labels(fname):
            with gzip.open(os.path.join(cache_dir, fname), 'rb') as f:
                return np.frombuffer(f.read(), np.uint8, offset=8)

        X_train = load_images(files['train_images'])
        y_train = load_labels(files['train_labels'])
        X_test  = load_images(files['test_images'])
        y_test  = load_labels(files['test_labels'])
    else:
        import tensorflow as tf
        if dataset_name == 'mnist':
            (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        else:
            (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], -1).astype(float) / 255.0
    X_test  = X_test.reshape(X_test.shape[0],  -1).astype(float) / 255.0
    return X_train, y_train.astype(int), X_test, y_test.astype(int)
