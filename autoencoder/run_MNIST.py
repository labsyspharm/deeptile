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
    cvae_model = CVAE.CVAE(latent_dim=50, input_shape=data_dict['data_shape'])
    optimizer = tf.keras.optimizers.Adam()
    ts_end = time.time()
    print('Prepare model took {:.3f} sec.'.format(ts_end-ts_start))
    # test run
    total_epoch = 20
    print('Start epoch loop ({} epochs in total).'.format(total_epoch))
    for epoch in range(1, total_epoch + 1):
        # train loop
        ts_start = time.time()
        train_loss = np.zeros(data_dict['train_batch_count'])
        for index, train_x in tqdm.tqdm(
                iterable=enumerate(data_dict['train_dataset']), 
                desc='train', 
                total=data_dict['train_batch_count'],
                disable=not verbose,
                ):
            train_loss[index] = CVAE.compute_apply_gradients(cvae_model, train_x, optimizer)
        train_elbo = -train_loss.mean()
        # progress report
        test_loss = np.zeros(data_dict['test_batch_count'])
        for index, test_x in tqdm.tqdm(
                iterable=enumerate(data_dict['test_dataset']), 
                desc='test', 
                total=data_dict['test_batch_count'],
                disable=not verbose,
                ):
            test_loss[index] = CVAE.compute_loss(cvae_model, test_x)
        test_elbo = -test_loss.mean()
        ts_end = time.time()
        print('Epoch: {}/{} done, Train ELBO: {:.3f} Test ELBO: {:.3f}, Runtime: {:.3f} sec.'\
                .format(epoch, total_epoch, train_elbo, test_elbo, ts_end-ts_start))
    print('Done.')
