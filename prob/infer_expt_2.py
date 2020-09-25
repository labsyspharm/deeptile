import numpy as np
from scipy import special, spatial
import itertools
import functools
import collections

from sklearn import mixture, preprocessing, decomposition

import tensorflow as tf
tfk = tf.keras
tfkl = tf.keras.layers

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

def generate_case(
        z, # embedded tiles
        y, # tile label
        state_0, # state before treatment
        state_1, # state after treatment
        size, # sample size for each state
        ):
    z_index = np.arange(z.shape[0]).astype(int)
    def generator():
        while True:
            z_0_index = stratefied_sampling(index=z_index, label=y, prob=state_0, size=size)
            z_1_index = stratefied_sampling(index=z_index, label=y, prob=state_1, size=size)
            yield (z[z_0_index, :], z[z_1_index, :])
    return generator()

def simulation_fn(
        generator_positive,
        generator_negative,
        n_components=2,
        n_tiles=int(1e3),
        n_cases=10,
        n_bootstrap=1000,
        ):
    # data
    positive_list = [next(generator_positive) for i in range(n_cases)]
    negative_list = [next(generator_negative) for i in range(n_cases)]
    all_z_state_0 = np.vstack([t[0] for t in positive_list+negative_list])
    all_z_state_1 = np.vstack([t[1] for t in positive_list+negative_list])
    all_z = np.vstack([all_z_state_0, all_z_state_1])

    # mixture model
    mixture_model = mixture.GaussianMixture(
            n_components=n_components,
            n_init=1,
            verbose=2,
            )
    mixture_model.fit(all_z)
    sortkey = np.argsort(mixture_model.means_[:, 0]) # for comparison

    # estimate states
    state_0_pred = []
    state_1_pred = []
    for case_tuple in positive_list + negative_list:
        z_state_0, z_state_1 = case_tuple
        state_0 = mixture_model.predict_proba(z_state_0)[:, sortkey].mean(axis=0)
        state_1 = mixture_model.predict_proba(z_state_1)[:, sortkey].mean(axis=0)
        state_0_pred.append(state_0)
        state_1_pred.append(state_1)
    state_0_pred = np.vstack(state_0_pred)
    state_1_pred = np.vstack(state_1_pred)

    # approximate transformation process with neural network
    # v0: (n_case, n_components)
    # v1: (n_case, n_components)
    # solve f(v0) -> v1
    treatment_model = tfk.Sequential(
        [tfkl.InputLayer(input_shape=(n_components,))]\
        + [tfkl.Dense(units=n_components, activation=tfk.activations.relu)] * 5\
        + [tfkl.Dense(units=n_components, activation=tfk.activations.softmax)])
    treatment_model.compile(optimizer=tfk.optimizers.Adam(), loss=tfk.losses.KLD)
    treatment_model.fit(x=state_0_pred, y=state_1_pred, verbose=0, epochs=20)
    optimal_loss = treatment_model.evaluate(x=state_0_pred, y=state_1_pred, verbose=0)

    # report
    result_dict = {
            'state_0_pred': state_0_pred,
            'state_1_pred': state_1_pred,
            'mixture_model': mixture_model,
            'sortkey': sortkey,
            'treatment_model': treatment_model,
            }

    return result_dict

if __name__ == '__main__':
    # load and preprocess MNIST embeddings from VAE
    z = np.load('./result/mnist_z.npy')
    y = np.load('./result/mnist_y.npy')
    z = preprocessing.scale(z)
    mask = functools.reduce(lambda x, y: x | y, [y == digit for digit in [0, 1, 2]])
    z = z[mask, ...]
    y = y[mask]

    # parameters
    params = {
            'n_components': 3,
            'n_tiles': 10000,
            'n_cases': 10,
            'n_bootstrap': 1000,
            }

    # get data generator
    generator_positive = generate_case(
        z=z, # embedded tiles
        y=y, # tile label
        state_0=np.array([0.1, 0.3, 0.6]), # state before treatment
        state_1=np.array([0.1, 0.0, 0.90]), # state after treatment
        size=params['n_tiles']) # sample size for each state
    generator_negative= generate_case(
        z=z, # embedded tiles
        y=y, # tile label
        state_0=np.array([0.0, 0.35, 0.65]), # state before treatment
        state_1=np.array([0.0, 0.4, 0.6]), # state after treatment
        size=params['n_tiles']) # sample size for each state

    result_dict = simulation_fn(
        generator_positive=generator_positive,
        generator_negative=generator_negative,
        **params,
        )
    print(result_dict['state_0_pred'])
    print(result_dict['state_1_pred'])

    if True:
        positive_start = [0.1, 0.3, 0.6]
        positive_end = [0.1, 0, 0.9]
        positive_pred = result_dict['treatment_model'].predict(np.array(positive_start).reshape((1,-1)))
        positive_pred = positive_pred[0, ...].tolist()
        negative_start = [0, 0.35, 0.65]
        negative_end = [0, 0.4, 0.6]
        negative_pred = result_dict['treatment_model'].predict(np.array(negative_start).reshape((1,-1)))
        negative_pred = negative_pred[0, ...].tolist()
        placeholder = ', '.join(['{:.2f}'] * len(positive_start))
        print('positive start [' + placeholder.format(*positive_start) + ']')
        print('positive end [' + placeholder.format(*positive_end) + ']')
        print('positive end [' + placeholder.format(*positive_pred) + '] (predicted)')
        print('negative start [' + placeholder.format(*negative_start) + ']')
        print('negative end [' + placeholder.format(*negative_end) + ']')
        print('negative end [' + placeholder.format(*negative_pred) + '] (predicted)')
