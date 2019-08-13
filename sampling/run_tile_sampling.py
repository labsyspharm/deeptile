import numpy as np
import pandas as pd
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

import deeptile_model
import deeptile_dataset
import deeptile_sampling

if __name__ == '__main__':
    # define tile shape
    displacement_cells = 10 # width/2, in unit of cells
    cell_size = 10 # in unit of micro-meter
    pixel_size = 0.65 # in unit of micro-meter
    tile_width = 2*int(displacement_cells * cell_size / pixel_size) # in unit of pixel
    tile_shape = (tile_width, tile_width)
    # paths
    image_filepath = '/n/scratch2/hungyiwu/deeptile_data/26531POST/'\
            'input_data/26531POST.ome.tif'
    channel_info_filepath = '/n/scratch2/hungyiwu/deeptile_data/26531POST/'\
            'input_data/channel_info.csv'
    workspace_folderpath = '/n/scratch2/hungyiwu/deeptile_data/26531POST/'\
            'output/workspace'
    record_filepath = './training_history.csv'
    # parse arguments
    parser = argparse.ArgumentParser(description='Get verbosity.')
    parser.add_argument('--verbose', action='store_true', # default is False
            help='Turn on tqdm progress bar.')
    args = parser.parse_args()
    verbose = args.verbose
    # data
    print('Initializing data loader...')
    ts_start = time.time()
    loader = deeptile_dataset.tile_loader(
            workspace_folderpath=workspace_folderpath,
            warm_start=True,
            image_filepath=image_filepath,
            channel_filepath=channel_info_filepath,
            )
    ts_end = time.time()
    print('Done in {:.3f} min.'.format((ts_end-ts_start)/60))
    # model
    cvae_model = deeptile_model.CVAE(
            latent_dim=20, 
            input_shape=tile_shape+(loader.count_channel,),
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
            )
    # training parameters
    batch_size = 10
    grid_count = 100
    # get grid dataset
    x_linspace = np.linspace(
            start=0, 
            stop=loader.image['image'].shape[0], 
            num=grid_count).astype(int)
    y_linspace = np.linspace(
            start=0, 
            stop=loader.image['image'].shape[1], 
            num=grid_count).astype(int)
    x_mesh, y_mesh = np.meshgrid(x_linspace, y_linspace)
    survey_grid = [(x,y) for x, y in zip(x_mesh.flatten(), y_mesh.flatten())]
    grid_dataset = loader.get_dataset(
            tile_shape=tile_shape,
            center_list=survey_grid,
            batch_size=batch_size,
            )
    # setup training, evaluation, sampling loop
    loss_list = []
    for train_x in tqdm.tqdm(grid_dataset, disable=not verbose):
        ts_start = time.time()
        loss = cvae_model.compute_apply_gradients(train_x)
        loss_list.append(loss.numpy())
        ts_end = time.time()
        print('batch runtime {:.3f} sec.'.format(ts_end-ts_start))
        break
    print(np.mean(loss_list))
    print('Done.')

'''
        ts_start = time.time()
        # phase 1: extract training sample
        # phase 2: train on sample
        for batch_tile in tqdm.tqdm(
                iterable=sample_dataset,
                desc='train',
                total=np.ceil(extractor.data_count/batch_size),
                disable=not verbose):
            cvae_model.compute_apply_gradients(batch_tile)
        # phase 3: survey on grid
        for batch_tile in tqdm.tqdm(
                iterable=grid_dataset,
                desc='survey',
                total=np.ceil(extractor.data_count/batch_size),
                disable=not verbose):
            loss = cvae_model.compute_loss(batch_tile)
            loss_list.append(loss.numpy())
        prob = np.array(loss_list)
        prob /= prob.sum() # note that regression loss >= 0
        # phase 4: generate next training sample
        train_sample = deeptile_sampling.multivariate_inverse_transform_sampling(
            data_X=survey_grid,
            data_prob=prob,
            sample_size=survey_grid.shape[0],
            )
        ts_end = time.time()
        # report progress
        mean_loss = np.mean(loss_list)
        runtime = ts_end-ts_start
        print('cycle {} done, mean loss {:.3f}, runtime {:.3f} sec.',format(
            cycle, mean_loss, runtime))
        # record history
        record.append([cycle, mean_loss, runtime])
    # save record to disk
    df = pd.DataFrame.from_records(record, columns=['cycle', 'avg. loss/tile', 'runtime (sec)'])
    df.to_csv(record_filepath)
    print('Done.')
'''
