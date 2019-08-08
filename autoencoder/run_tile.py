import numpy as np
import time
import argparse

import tqdm
import tensorflow as tf

# set default tensor precision
tf.keras.backend.set_floatx('float32')
# turn on memory growth so allocation is as-needed
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

import load_tile_dataset
import CVAE

if __name__ == '__main__':
    # get verbose flag
    parser = argparse.ArgumentParser(description='Get verbose flag.')
    parser.add_argument('--verbose', action='store_true') # default is False
    verbose = parser.parse_args().verbose
    # load dataset
    ts_start = time.time()
    data_dict = load_tile_dataset.load(batch_size=10)
    ts_end = time.time()
    print('Prepare dataset took {:.3f} sec.'.format(ts_end-ts_start))
    # setup model and optimizer
    ts_start = time.time()
    cvae_model = CVAE.CVAE(
            latent_dim=10, 
            input_shape=data_dict['data_shape'],
            optimizer=tf.keras.optimizers.Adam(),
            )
    ts_end = time.time()
    print('Prepare model took {:.3f} sec.'.format(ts_end-ts_start))
    # epoch loop
    total_epoch = 5
    print('Start epoch loop ({} epochs in total).'.format(total_epoch))
    for epoch in range(1, total_epoch + 1):
        # train loop
        ts_start = time.time()
        train_loss = np.zeros(data_dict['train_batch_count'])
        for index, (train_x,) in tqdm.tqdm(
                iterable=enumerate(data_dict['train_dataset']), 
                desc='train', 
                total=data_dict['train_batch_count'],
                disable=not verbose,
                ):
            loss = cvae_model.compute_apply_gradients(train_x)
            train_loss[index] = loss.numpy()
        train_elbo = -train_loss.mean()
        # progress report
        test_loss = np.zeros(data_dict['test_batch_count'])
        for index, (test_x,) in tqdm.tqdm(
                iterable=enumerate(data_dict['test_dataset']), 
                desc='test',
                total=data_dict['test_batch_count'],
                disable=not verbose,
                ):
            loss = cvae_model.compute_loss(test_x)
            test_loss[index] = loss.numpy()
        test_elbo = -test_loss.mean()
        ts_end = time.time()
        print('Epoch: {}/{} done, Train ELBO: {:.3f} Test ELBO: {:.3f}, Runtime: {:.3f} sec.'\
                .format(epoch, total_epoch, train_elbo, test_elbo, ts_end-ts_start))
    print('Done.')

