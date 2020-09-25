import os
import shutil

import numpy as np
import time
import argparse

import tqdm
import tensorflow as tf

import load_MNIST_dataset
import CVAE

if __name__ == '__main__':
    # get verbose flag
    parser = argparse.ArgumentParser(description='Get verbose flag.')
    parser.add_argument('--verbose', action='store_true') # default is False
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--latent_dim', type=int, default=5)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--log10_lr', type=float, default=-5.0)
    parser.add_argument('--out', type=str, default='weight')
    args = parser.parse_args()

    VERBOSE = args.verbose
    BATCH_SIZE = args.batch_size
    LATENT_DIM = args.latent_dim
    TOTAL_EPOCH = args.epoch
    LEARNING_RATE = 10**(args.log10_lr)
    output_name = args.out

    # get dataset
    ts_start = time.time()
    data_dict= load_MNIST_dataset.load(batch_size=BATCH_SIZE)
    ts_end = time.time()
    print('Prepare dataset took {:.3f} sec.'.format(ts_end-ts_start), flush=True)

    # setup model and optimizer
    ts_start = time.time()
    cvae_model = CVAE.CVAE(
            latent_dim=LATENT_DIM, 
            input_shape=data_dict['data_shape'],
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            )
    ts_end = time.time()
    print('Prepare model took {:.3f} sec.'.format(ts_end-ts_start), flush=True)

    # utility functions
    def train():
        ts_start = time.time()
        train_loss = []
        for index, train_x in tqdm.tqdm(
                iterable=enumerate(data_dict['train_dataset']), 
                desc='train', 
                total=data_dict['train_batch_count'],
                disable=not VERBOSE,
                ):
            loss = cvae_model.compute_apply_gradients(train_x)
            train_loss.append(loss.numpy())
        train_elbo = -np.mean(train_loss)
        ts_end = time.time()
        return dict(elbo=train_elbo, runtime=ts_end-ts_start)

    def test():
        ts_start = time.time()
        test_loss = []
        for index, test_x in tqdm.tqdm(
                iterable=enumerate(data_dict['test_dataset']), 
                desc='test', 
                total=data_dict['test_batch_count'],
                disable=not VERBOSE,
                ):
            loss = cvae_model.compute_loss(test_x)
            test_loss.append(loss.numpy())
        test_elbo = -np.mean(test_loss)
        ts_end = time.time()
        return dict(elbo=test_elbo, runtime=ts_end-ts_start)

    # main loop
    test_output = test()
    print('before training, ELBO: {:.3f}, runtime: {:.0f} sec'\
            .format(test_output['elbo'], test_output['runtime']), flush=True)
    for epoch in range(1, TOTAL_EPOCH + 1):
        train_output = train()
        test_output = test()
        print('epoch: {}/{}'.format(epoch, TOTAL_EPOCH), flush=True)
        print('train ELBO: {:.3f}, runtime: {:.0f} sec'\
                .format(train_output['elbo'], train_output['runtime']), flush=True)
        print('test ELBO: {:.3f}, runtime: {:.0f} sec'\
                .format(test_output['elbo'], test_output['runtime']), flush=True)
        # early-stop detection
        if np.isnan(train_output['elbo']) or np.isnan(test_output['elbo']):
            print('NaN detected, train loop terminated.', flush=True)
            break
    print('Done.', flush=True)

    # save model
    output_folderpath = './{}'.format(output_name)
    if os.path.isdir(output_folderpath):
        shutil.rmtree(output_folderpath)
    os.mkdir(output_folderpath)
    cvae_model.save_weights(os.path.join(output_folderpath, 'weights'))
