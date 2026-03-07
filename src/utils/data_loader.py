import numpy as np
import tensorflow as tf


def load_dataset(dataset_name):
    if dataset_name == 'mnist':
        (X_train,y_train),(X_test,y_test) = tf.keras.datasets.mnist.load_data()
    else:
        (X_train,y_train),(X_test,y_test) = tf.keras.datasets.fashion_mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0],-1) / 255.0
    X_test = X_test.reshape(X_test.shape[0],-1) / 255.0
    return X_train,y_train,X_test,y_test


def one_hot_encode(y,num_classes=10):
    m = y.shape[0]
    Y = np.zeros((m,num_classes))
    for i in range(m):
        Y[i,y[i]] = 1
    return Y
