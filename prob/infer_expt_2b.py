import numpy as np
import os
import itertools
import functools
from scipy import stats

from sklearn import cluster, preprocessing, metrics

def stratefied_sampling(index, label, prob, size):
    sample = np.zeros(size).astype(index.dtype)
    label_count = (prob * size).astype(int)
    progress_index = 0
    for label_i in range(prob.shape[0]):
        if label_count[label_i] > 0:
            index_i = index[label == label_i]
            index_i_sample = np.random.choice(index_i, size=label_count[label_i], replace=True)
            sample[progress_index:progress_index+label_count[label_i]] = index_i_sample
            progress_index += label_count[label_i]
    return sample

def infer(z, n_components):
    # clustering
    km_model = cluster.MiniBatchKMeans(n_clusters=n_components)
    km_model.fit(z)
    y_pred = km_model.predict(z)
    anchor_pred = km_model.cluster_centers_[:, 0]
    return y_pred, anchor_pred

def match_digit(digits_used, anchor, y_pred, anchor_pred):
    y_pred_matched = np.zeros(y_pred.shape).astype(digits_used.dtype)
    for y_pred_i, rank in enumerate(stats.rankdata(anchor_pred)):
        digit = digits_used[np.argsort(anchor)[int(rank)-1]]
        y_pred_matched[y_pred == y_pred_i] = digit
    return y_pred_matched

if __name__ == '__main__':
    # load and preprocess MNIST embeddings from VAE
    folderpath = '/n/scratch2/hungyiwu/deeptile/result/'
    z = np.load(os.path.join(folderpath, 'mnist_z.npy'))
    y = np.load(os.path.join(folderpath, 'mnist_y.npy'))
    z = preprocessing.scale(z)
    digits_used = np.array([0, 1, 2])
    mask = functools.reduce(lambda x, y: x | y, [y == digit for digit in digits_used])
    z = z[mask, ...]
    y = y[mask]
    n_components = 3
    dim = z.shape[1]
    # test
    prob_list = [[0.33, 0.33, 0.34], [0.1, 0.3, 0.6]]
    for prob in prob_list:
        # sampling
        true_prob = np.array(prob)
        index = stratefied_sampling(
                index=np.arange(z.shape[0]).astype(int),
                label=y,
                prob=true_prob,
                size=1000)
        np.random.shuffle(index)
        z_sample = z[index, ...]
        y_sample = y[index]
        # inference
        y_pred, anchor_pred = infer(z_sample, n_components)
        anchor = np.array([z_sample[y_sample == digit].mean(axis=0)[0] for digit in digits_used])
        y_pred_matched = match_digit(digits_used, anchor, y_pred, anchor_pred)
        # evaluate
        print('true prob:', true_prob)
        print('average:', np.mean(y_pred_matched == y_sample))
        print('y (first 10)', y_sample[0:10])
        print('y_pred (first 10)', y_pred[0:10])
        print('y_matched (first 10)', y_pred_matched[0:10])
        print('confusion matrix')
        print(metrics.confusion_matrix(y_sample, y_pred_matched))
        print('=' * 20)
