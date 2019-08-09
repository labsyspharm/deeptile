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
    verbose = parser.parse_args().verbose
    # get dataset
    ts_start = time.time()
    BATCH_SIZE = 100
    data_dict= load_MNIST_dataset.load(batch_size=BATCH_SIZE)
    ts_end = time.time()
    print('Prepare dataset took {:.3f} sec.'.format(ts_end-ts_start))
    # setup model and optimizer
    ts_start = time.time()
    cvae_model = CVAE.CVAE(
            latent_dim=50, 
            input_shape=data_dict['data_shape'],
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            )
    ts_end = time.time()
    print('Prepare model took {:.3f} sec.'.format(ts_end-ts_start))
    # test run
    total_epoch = 10
    print('Start epoch loop ({} epochs in total).'.format(total_epoch))
    for epoch in range(1, total_epoch + 1):
        # train loop
        ts_start = time.time()
        train_loss = []
        for index, train_x in tqdm.tqdm(
                iterable=enumerate(data_dict['train_dataset']), 
                desc='train', 
                total=data_dict['train_batch_count'],
                disable=not verbose,
                ):
            loss = cvae_model.compute_apply_gradients(train_x)
            train_loss.append(loss.numpy())
        train_elbo = -np.mean(train_loss)
        # progress report
        test_loss = []
        for index, test_x in tqdm.tqdm(
                iterable=enumerate(data_dict['test_dataset']), 
                desc='test', 
                total=data_dict['test_batch_count'],
                disable=not verbose,
                ):
            loss = cvae_model.compute_loss(test_x)
            test_loss.append(loss.numpy())
        test_elbo = -np.mean(test_loss)
        ts_end = time.time()
        print('Epoch: {}/{} done, Train ELBO: {:.3f} Test ELBO: {:.3f}, Runtime: {:.3f} sec.'\
                .format(epoch, total_epoch, train_elbo, test_elbo, ts_end-ts_start))
        if np.isnan(train_elbo) or np.isnan(test_elbo):
            print('NaN detected, train loop terminated.')
            break
    print('Done.')
