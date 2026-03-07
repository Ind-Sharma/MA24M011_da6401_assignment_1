import numpy as np
import gzip
import os

def load_dataset(dataset_name):
    folder = 'mnist' if dataset_name=='mnist' else 'fashion-mnist'
    base = os.path.join(os.path.expanduser('~'),'.keras','datasets',folder)
    fnames = {
        'tri': 'train-images-idx3-ubyte.gz',
        'trl': 'train-labels-idx1-ubyte.gz',
        'tei': 't10k-images-idx3-ubyte.gz',
        'tel': 't10k-labels-idx1-ubyte.gz',
    }
    if all(os.path.exists(os.path.join(base,f)) for f in fnames.values()):
        def read_imgs(fn):
            with gzip.open(os.path.join(base,fn),'rb') as f:
                return np.frombuffer(f.read(),np.uint8,offset=16).reshape(-1,28,28)
        def read_lbls(fn):
            with gzip.open(os.path.join(base,fn),'rb') as f:
                return np.frombuffer(f.read(),np.uint8,offset=8)
        x1 = read_imgs(fnames['tri'])
        y1 = read_lbls(fnames['trl'])
        x2 = read_imgs(fnames['tei'])
        y2 = read_lbls(fnames['tel'])
    else:
        import tensorflow as tf
        if dataset_name=='mnist':
            (x1,y1),(x2,y2) = tf.keras.datasets.mnist.load_data()
        else:
            (x1,y1),(x2,y2) = tf.keras.datasets.fashion_mnist.load_data()
    x1 = x1.reshape(x1.shape[0],-1).astype(float)/255.0
    x2 = x2.reshape(x2.shape[0],-1).astype(float)/255.0
    return x1,y1.astype(int),x2,y2.astype(int)
