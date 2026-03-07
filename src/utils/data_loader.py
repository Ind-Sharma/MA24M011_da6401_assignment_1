import numpy as np


def load_dataset(dataset_name):
    from sklearn.datasets import fetch_openml
    import tensorflow as tf

    if dataset_name == 'mnist':
        try:
            data = fetch_openml('mnist_784',version=1,as_frame=False,parser='auto')
        except TypeError:
            data = fetch_openml('mnist_784',version=1,as_frame=False)
        X = data.data.astype(float) / 255.0
        y = data.target.astype(int)
        X_train,y_train = X[:60000],y[:60000]
        X_test,y_test = X[60000:],y[60000:]
    else:
        (X_train,y_train),(X_test,y_test) = tf.keras.datasets.fashion_mnist.load_data()
        X_train = X_train.reshape(X_train.shape[0],-1) / 255.0
        X_test = X_test.reshape(X_test.shape[0],-1) / 255.0
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

    return X_train,y_train,X_test,y_test
