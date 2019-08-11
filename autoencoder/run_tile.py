import numpy as np
import time
import argparse
import os

import tqdm
import tensorflow as tf
'''
# set default tensor precision
tf.keras.backend.set_floatx('float32')
# turn on memory growth so allocation is as-needed
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
'''
import load_tile_dataset
import CVAE

if __name__ == '__main__':
    # get verbose flag
    parser = argparse.ArgumentParser(description='Get verbose flag.')
    parser.add_argument('--verbose', action='store_true') # default is False
    verbose = parser.parse_args().verbose
    # load dataset
    ts_start = time.time()
    data_dict = load_tile_dataset.load(
            batch_size=10,
            train_fraction=0.084, # ~12k, eta 2 days
            test_fraction=0.0068, # ~1k
            )
    ts_end = time.time()
    print('Prepare dataset took {:.3f} sec.'.format(ts_end-ts_start))
    # setup model and optimizer
    ts_start = time.time()
    cvae_model = CVAE.CVAE(
            latent_dim=20, 
            input_shape=data_dict['data_shape'],
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
            )
    ts_end = time.time()
    print('Prepare model took {:.3f} sec.'.format(ts_end-ts_start))
    # train/fit model on data
    total_epoch =100
    print('Start epoch loop ({} epochs in total).'.format(total_epoch))
    for epoch in range(1, total_epoch + 1):
        # training
        ts_start = time.time()
        train_loss = []
        for (train_x,) in tqdm.tqdm(
                iterable=data_dict['train_dataset'],
                desc='train', 
                total=data_dict['train_batch_count'],
                disable=not verbose,
                ):
            loss = cvae_model.compute_apply_gradients(train_x)
            train_loss.append(loss.numpy())
        train_elbo = -np.mean(train_loss)
        # evaluation
        test_loss = []
        for (test_x,) in tqdm.tqdm(
                iterable=data_dict['test_dataset'],
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
    # save embedding for later analysis
    output_folderpath = '/n/scratch2/hungyiwu/deeptile_data/26531POST/output/'
    embedding_filename = 'embedding.npy'
    embedding_filepath = os.path.join(output_folderpath, embedding_filename)
    embedding_list = []
    for (test_x,) in tqdm.tqdm(
            iterable=data_dict['test_dataset'],
            desc='embed',
            total=data_dict['test_batch_count'],
            disable=not verbose,
            ):
        mean, logvar = cvae_model.encode(train_x)
        z = cvae_model.reparameterize(mean, logvar)
        embedding_list.append(z.numpy())
    embedding = np.concatenate(embedding_list, axis=0)
    np.save(embedding_filepath, embedding)
    print('Done.')

