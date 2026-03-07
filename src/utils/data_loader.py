import numpy as np


def load_dataset(dataset_name):
    import tensorflow as tf
    if dataset_name == 'mnist':
        (X_train,y_train),(X_test,y_test) = tf.keras.datasets.mnist.load_data()
    else:
        (X_train,y_train),(X_test,y_test) = tf.keras.datasets.fashion_mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0],-1).astype(float) / 255.0
    X_test = X_test.reshape(X_test.shape[0],-1).astype(float) / 255.0
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    return X_train,y_train,X_test,y_test
