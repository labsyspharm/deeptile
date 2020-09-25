import numpy as np
import os
import itertools
import functools

from sklearn import mixture, preprocessing
import tqdm

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

if __name__ == '__main__':
    # load and preprocess MNIST embeddings from VAE
    folderpath = '/n/scratch2/hungyiwu/deeptile/result/'
    z = np.load(os.path.join(folderpath, 'mnist_z.npy'))
    y = np.load(os.path.join(folderpath, 'mnist_y.npy'))
    z = preprocessing.scale(z)
    mask = functools.reduce(lambda x, y: x | y, [y == digit for digit in [0, 1, 2]])
    z = z[mask, ...]
    y = y[mask]
    n_components = 3
    dim = z.shape[1]

    prob_dict = {
            'balanced_prob': np.array([0.3, 0.3, 0.4]),
            'imbalanced_prob': np.array([0.1, 0.3, 0.6]),
            }
    for context in prob_dict:
        print('=== {} ==='.format(context))
        print('true prob', prob_dict[context])
        # stratefied sampling
        index = stratefied_sampling(
                index=np.arange(z.shape[0]).astype(int),
                label=y,
                prob=prob_dict[context],
                size=1000)

        # positive control
        print('\ntrained on full data')
        m = mixture.GaussianMixture(
                n_components=n_components,
                covariance_type='diag',
                )
        m.fit(z)
        full_data_params = {
                'weights_init': m.weights_,
                'means_init': m.means_,
                'precisions_init': m.precisions_,
                }
        sortkey = np.argsort(m.means_[:, 0])
        print(m.predict_proba(z[index, :]).mean(axis=0)[sortkey])
        print('log likelihood', m.score(z[index, :]))

        # negative control
        print('\ntrained on sample')
        m = mixture.GaussianMixture(
                n_components=n_components,
                covariance_type='diag',
                )
        m.fit(z[index, :])
        sortkey = np.argsort(m.means_[:, 0])
        print(m.predict_proba(z[index, :]).mean(axis=0)[sortkey])
        print('log likelihood', m.score(z[index, :]))

        # rescue positive control
        print('\ntrained on sample, initialized on params trained on full data')
        m = mixture.GaussianMixture(
                n_components=n_components,
                covariance_type='diag',
                **full_data_params,
                )
        m.fit(z[index, :])
        sortkey = np.argsort(m.means_[:, 0])
        print(m.predict_proba(z[index, :]).mean(axis=0)[sortkey])
        print('log likelihood', m.score(z[index, :]))

        # experiment group
        print('\ntrained on sample, grid search')
        count = 20
        axis = np.linspace(0, 1, count)
        max_LL = -np.inf
        optimal_w = None
        for t in itertools.product(axis, repeat=(n_components-1)):
            # get weight initializer
            w_last = n_components - sum(t)
            pk = np.zeros(n_components)
            pk[0:-1] = t
            pk[-1] = w_last
            pk += np.finfo(float).eps # for numerical stability
            pk /= pk.sum()
            # calculate log likelihood (LL)
            m = mixture.GaussianMixture(
                    n_components=n_components,
                    covariance_type='diag',
                    weights_init=pk,
                    )
            m.fit(z[index, :])
            current_LL = m.score(z[index, :])
            # update
            if current_LL > max_LL:
                optimal_w = pk
                max_LL = current_LL
        # evaluate optimal model
        m = mixture.GaussianMixture(
                n_components=n_components,
                covariance_type='diag',
                weights_init=optimal_w,
                )
        m.fit(z[index, :])
        sortkey = np.argsort(m.means_[:, 0])
        print(m.predict_proba(z[index, :]).mean(axis=0)[sortkey])
        print('log likelihood', m.score(z[index, :]))
        print()
