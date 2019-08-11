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
    # define tile size
    displacement_cells = 10 # width/2, in unit of cells
    cell_size = 10 # in unit of micro-meter
    pixel_size = 0.65 # in unit of micro-meter
    tile_width = 2*int(displacement_cells * cell_size / pixel_size) # in unit of pixel
    # paths
    image_filepath = '/n/scratch2/hungyiwu/deeptile_data/26531POST/input_data/26531POST.ome.tif'
    channel_info_filepath = '/n/scratch2/hungyiwu/deeptile_data/26531POST/input_data/channel_info.csv'
    sample_folderpath = '/n/scratch2/hungyiwu/deeptile_data/26531POST/output/tile_train_sample'
    grid_folderpath = '/n/scratch2/hungyiwu/deeptile_data/26531POST/output/tile_survey_grid'
    record_filepath = './training_history.csv'
    # parse arguments
    parser = argparse.ArgumentParser(description='Get verbose flag and CPU count.')
    parser.add_argument('--verbose', action='store_true', # default is False
            help='Turn on tqdm progress bar.')
    parser.add_argument('-n', type=int, default=1,
            help='CPU count for parallel data processing (default: 1).')
    args = parser.parse_args()
    verbose = args.verbose
    cpu_count = args.n
    # setup model
    extractor = deeptile_dataset.tile_extractor(
            image_filepath=image_filepath,
            channel_info_filepath=channel_info_filepath,
            tile_width=tile_width,
            cpu_count=cpu_count,
            )
    sample_loader = deeptile_dataset.tile_loader(
            tile_folderpath=sample_folderpath,
            data_shape=extractor.data_shape,
            )
    grid_loader = deeptile_dataset.tile_loader(
            tile_folderpath=grid_folderpath,
            data_shape=extractor.data_shape,
            )
    cvae_model = deeptile_model.CVAE(
            latent_dim=20, 
            input_shape=extractor.data_shape,
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
            )
    # training parameters
    batch_size = 10
    total_cycle = 5
    grid_count = 100
    # get grid dataset
    x_linspace = np.linspace(start=0, stop=extractor.image_shape[0], num=grid_count).astype(int)
    y_linspace = np.linspace(start=0, stop=extractor.image_shape[1], num=grid_count).astype(int)
    x_mesh, y_mesh = np.meshgrid(x_linspace, y_linspace)
    survey_grid = np.vstack([x_mesh.flatten(), y_mesh.flatten()]).T
    extractor.extract(
            sample_X=survey_grid,
            output_folderpath=grid_folderpath,
            )
    grid_dataset = grid_loader.get_dataset(batch_size=batch_size)
    # setup training, evaluation, sampling loop
    train_sample = survey_grid.copy()
    record = []
    for cycle in range(total_cycle):
        ts_start = time.time()
        # phase 1: train on sample
        extractor.extract(
                sample_X=train_sample,
                output_folderpath=tile_folderpath,
                )
        sample_dataset = sample_loader.get_dataset(batch_size=batch_size)
        for batch_tile in sample_dataset:
            cvae_model.compute_apply_gradients(batch_tile)
        # phase 2: survey on grid
        loss_list = []
        for batch_tile in grid_dataset:
            loss = cvae_model.compute_loss(batch_tile)
            loss_list.append(loss.numpy())
        prob = np.array(loss_list)
        prob /= prob.sum() # note that regression loss >= 0
        # phase 3: generate next training sample
        train_sample = deeptile_sampling.multivariate_inverse_transform_sampling(
            data_X=survey_grid,
            data_prob=prob,
            sample_size=survey_grid.shape[0],
            )
        ts_end = time.time()
        # record history
        record.append([cycle, np.mean(loss_list), ts_end-ts_start])
    # save record to disk
    df = pd.DataFrame.from_records(record, columns=['cycle', 'avg. loss/tile', 'runtime (sec)'])
    df.to_csv(record_filepath)
    print('Done.')

