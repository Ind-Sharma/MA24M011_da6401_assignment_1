import numpy as np


def _load_from_keras_cache(dataset_name):
    import gzip, os
    folder = 'mnist' if dataset_name == 'mnist' else 'fashion-mnist'
    cache_dir = os.path.join(os.path.expanduser('~'), '.keras', 'datasets', folder)
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images':  't10k-images-idx3-ubyte.gz',
        'test_labels':  't10k-labels-idx1-ubyte.gz',
    }
    if not all(os.path.exists(os.path.join(cache_dir, f)) for f in files.values()):
        return None

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
    return (X_train, y_train), (X_test, y_test)


def load_dataset(dataset_name):
    loaded = False

    # 1. Read directly from keras/TF on-disk cache (fastest, no imports needed)
    try:
        result = _load_from_keras_cache(dataset_name)
        if result is not None:
            (X_train, y_train), (X_test, y_test) = result
            loaded = True
    except Exception:
        pass

    # 2. Try sklearn with liac-arff parser (no pandas needed)
    if not loaded and dataset_name == 'mnist':
        try:
            from sklearn.datasets import fetch_openml
            data = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
            X = data.data.astype(float)
            y = data.target.astype(int)
            X_train, X_test = X[:60000].reshape(-1, 28, 28), X[60000:].reshape(-1, 28, 28)
            y_train, y_test = y[:60000], y[60000:]
            loaded = True
        except Exception:
            pass

    # 3. tensorflow.keras
    if not loaded:
        import tensorflow as tf
        if dataset_name == 'mnist':
            (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        else:
            (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], -1).astype(float) / 255.0
    X_test  = X_test.reshape(X_test.shape[0],  -1).astype(float) / 255.0
    return X_train, y_train.astype(int), X_test, y_test.astype(int)
