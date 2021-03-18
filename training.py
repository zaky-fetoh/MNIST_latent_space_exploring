import keras.callbacks as callbacks
import keras.datasets as datasets
import keras.utils as utils
import numpy as np
import model

reshape = lambda X: X[..., np.newaxis]
normalize = lambda X: X / 127.5 - 1


def get_mnist_data(dataset=datasets.mnist.load_data,
                   normalize=normalize,
                   categ=utils.to_categorical,
                   reshape=reshape
                   ):
    compose = lambda X: normalize(reshape(X))
    (X_tr, Y_tr), (X_ts, Y_ts) = dataset()
    X_tr, X_ts = [compose(X) for X in [X_tr, X_ts]]
    Y_tr, Y_ts = [categ(X) for X in [Y_tr, Y_ts]]
    return [X_tr, Y_tr], [X_ts, Y_ts]

if __name__ == '__main__':
    ae = model.compile_auto_enco()
    filepath = 'waight{epoch:04d}.h5'
    [X_tr, Y_tr], [X_ts, Y_ts] = get_mnist_data()
    ae.fit(X_tr, X_tr,124, 20,
           callbacks=[callbacks.ModelCheckpoint(filepath)])