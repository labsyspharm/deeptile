import numpy as np
import pandas as pd
import time
import argparse
import functools
import os
import pickle

import skimage.transform
import tensorflow as tf
import tqdm
import shutil

import ACAE2

# turn on memory growth so GPU memory allocation becomes as-needed
# for cases when training takes too much memory.
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

def my_perturbation(prob_before):
    prob_after = np.zeros(prob_before.shape)
    for i in range(prob_before.shape[0]):
        noise = np.random.randn() * 0.02
        if i in [0, 1, 2, 3]:
            prob_after[i] = prob_before[i] + noise
        elif i in [4, 5, 6]:
            prob_after[i] = prob_before[i] * 1.5 + noise
        elif i in [7, 8, 9]:
            prob_after[i] = prob_before[i] * 0.5 + noise
    prob_after[prob_after < 0] = 0
    prob_after /= prob_after.sum()
    return prob_after

def prob_sampling(y, prob, n):
    index = np.arange(y.shape[0]).astype(int)
    sample_list = []
    for digit in range(prob.shape[0]):
        sample_count = int(n * prob[digit])
        sample_index = np.random.choice(
                index[y == digit], 
                size=sample_count,
                replace=True,
                )
        sample_list.append(sample_index)
    sample_index = np.hstack(sample_list)
    return sample_index

def get_mnist_generator(x, index):
    def mnist_generator():
        for i in index:
            yield (x[i, ...],)
    return mnist_generator

def get_dataset_lists(patient_count, tile_count, dirichlet_prior, x, y, batch_size):
    dataset_list_pre = []
    dataset_list_post = []
    for patient_index in range(patient_count):
        prob_pre = np.random.dirichlet(alpha=dirichlet_prior, size=1).flatten()
        prob_post = my_perturbation(prob_pre)
        x_index_pre = prob_sampling(y, prob_pre, tile_count)
        x_index_post = prob_sampling(y, prob_post, tile_count)
        x_pre_dataset = tf.data.Dataset.from_generator(
                generator=get_mnist_generator(x, x_index_pre),
                output_types=(np.float32,),
                ).shuffle(tile_count)\
                .batch(batch_size, drop_remainder=True)
        x_post_dataset = tf.data.Dataset.from_generator(
                generator=get_mnist_generator(x, x_index_post),
                output_types=(np.float32,),
                ).shuffle(tile_count)\
                .batch(batch_size, drop_remainder=True)
        dataset_list_pre.append(x_pre_dataset)
        dataset_list_post.append(x_post_dataset)
    return dataset_list_pre, dataset_list_post

if __name__ == '__main__':
    # parameters
    BATCH_SIZE = 64
    LATENT_DIM = 20
    TOTAL_EPOCH = int(1e1)
    DIRICHLET_PRIOR = np.ones(10)
    TOTAL_PATIENT = 10
    TILE_PER_PATIENT = int(1e3)
    LEARNING_RATE = 1e-6
    # parse arguments
    parser = argparse.ArgumentParser(description='Get verbosity.')
    parser.add_argument('--verbose', action='store_true', # default is False
            help='Turn on tqdm progress bar.')
    parser.add_argument('--jobid', help='Slurm job id.', default='default')
    args = parser.parse_args()
    # derived parameters
    output_folderpath = 'joboutput_'+args.jobid
    train_history_filepath = os.path.join(output_folderpath, 'train_history.csv')
    model_filepath = os.path.join(output_folderpath, 'model.pkl')
    if os.path.isdir(output_folderpath):
        shutil.rmtree(output_folderpath)
    os.makedirs(output_folderpath)
    # data preprocessing
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype(np.float32) # type coersion
    x_test = x_test.astype(np.float32)
    x_train /= 255 # normalize to [0, 1]
    x_test /= 255
    x_train = np.stack([x_train]*3, axis=-1) # pseudo-coloring
    x_test = np.stack([x_test]*3, axis=-1)
    # get samples
    train_dataset_list_pre, train_dataset_list_post = get_dataset_lists(
            patient_count=TOTAL_PATIENT,
            tile_count=TILE_PER_PATIENT,
            dirichlet_prior=DIRICHLET_PRIOR,
            x=x_train,
            y=y_train,
            batch_size=BATCH_SIZE,
            )
    test_dataset_list_pre, test_dataset_list_post = get_dataset_lists(
            patient_count=TOTAL_PATIENT,
            tile_count=TILE_PER_PATIENT,
            dirichlet_prior=DIRICHLET_PRIOR,
            x=x_test,
            y=y_test,
            batch_size=BATCH_SIZE,
            )
    # model and optimizer
    model = ACAE2.ACAE(latent_dim=LATENT_DIM, feature_shape=x_train[0].shape)
    opt = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
    # loop
    for epoch in tqdm.tqdm(
            iterable=range(TOTAL_EPOCH),
            desc='main loop',
            disable=not args.verbose,
            ):
        ts_start = time.time()
        train_embedding_loss, train_perturbation_loss = model.compute_apply_gradients(
                train_dataset_list_pre, train_dataset_list_post, opt)
        test_embedding_loss, test_perturbation_loss = model.compute_loss(
                test_dataset_list_pre, test_dataset_list_post)
        ts_end = time.time()
        record = [epoch, ts_end-ts_start,
                train_embedding_loss, train_perturbation_loss, 
                test_embedding_loss, test_perturbation_loss,
                ]
        print('epoch {}, runtime {:.3f} sec, \
                train embedding loss {:.3E}, train perturbation loss {:.3E}, \
                test embedding loss {:.3E}, test perturbation loss {:.3E}'.format(*record), 
                flush=True)

