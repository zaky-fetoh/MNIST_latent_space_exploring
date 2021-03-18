import matplotlib.pyplot as plt
import training as data
import numpy as np
import model

find_feature = lambda encoder, X: encoder.predict(X)
normalize = lambda x: (x - x.min()) / (x.max() - x.min() + 1)
scale = lambda X, s: X * (np.array(s)[np.newaxis, ...])
trunk = lambda x, Min, Max: min(max(x, Min), Max)
entropy = lambda p: -np.sum(p * np.log(p))




def same_plane_plting(feature, Y, title='ploting test data'):
    for i in range(10):
        mask = Y == i
        plt.plot(feature[mask][:, 0], feature[mask][:, 1],
                 '+', label=str(i))
    plt.legend()
    plt.title(title)


plting = lambda encoder, X, Y, title: \
    same_plane_plting(find_feature(encoder, X), Y, title)


def get_channel_mapped_class_array(feature, Y,
                                   n_class=10,
                                   resol=(1000, 1000),
                                   normalize=normalize,
                                   scale=scale,
                                   trunk=lambda x: x):
    class_freq = np.zeros((n_class,) + resol, dtype=np.float)
    prepro = lambda x: trunk(scale(normalize(x), resol))
    feature = prepro(feature).astype(np.int)
    for i in range(n_class):
        coor = feature[Y == i]
        for j in range(coor.shape[0]):
            class_freq[i, coor[j, 0], coor[j, 1]] += 1
    return class_freq


short_get_channel_mapped_class_array = lambda encoder, X, Y: \
    get_channel_mapped_class_array(find_feature(encoder, X), Y)


def plot_mapped_channel(class_channel, n_class=10):
    for i in range(n_class):
        plt.figure()
        plt.imshow(class_channel[i], cmap='seismic')

def moving_entropy(class_freq,
                   win_size=(3, 3),
                   entr=entropy):
    clss, row, col = class_freq.shape
    rr, cc = [x // 2 for x in win_size]
    entr_array = np.zeros((row, col), dtype=np.float)
    for i in range(row):
        for j in range(col):
            crop = class_freq[:, i - rr:i + rr, j - cc:j + cc].copy()
            crop = np.reshape(crop, (clss, -1))
            p = np.sum(crop, -1) / np.sum(crop)
            entr_array[i, j] = entr(p[p>0])
    return np.nan_to_num(entr_array)



if __name__ == "__main__":
    enco = model.get_trained_model().layers[0]
    [X_tr, Y_tr], [X_ts, Y_ts] = data.get_mnist_data(categ=lambda x: x)
    # plting(enco, X_ts, Y_ts, 'ploting_train_encoded_traing_data')
    feature = find_feature(enco, X_tr)
    cass_freq = get_channel_mapped_class_array(feature, Y_tr, resol=(100, 100))
    entr_arr = moving_entropy(cass_freq)
    plt.imshow(normalize(entr_arr), cmap='seismic')