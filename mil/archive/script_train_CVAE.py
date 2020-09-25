import numpy as np
import pandas as pd
import time
import argparse
import os
import copy
import yaml

import tqdm
import tensorflow as tf

import deeptile_model
import deeptile_dataset
import deeptile_sampling

if __name__ == '__main__':
    # parameter defined within the scope of model preparation
    BATCH_SIZE = 10
    LATENT_DIM = 20
    LEARNING_RATE = 1e-5
    GRID_NUM = 1000
    TOTAL_CYCLE = 100
    EPOCH_PER_CYCLE = 10
    # load configuration
    workspace_folderpath = '/n/scratch2/hungyiwu/deeptile_data/26531POST/output/workspace'
    config_filepath = os.path.join(workspace_folderpath, 'default_config.yaml')
    with open(config_filepath, 'r') as yaml_file:
        config_dict = yaml.safe_load(yaml_file)
    # paths
    history_filepath = os.path.join(config_dict['workspace_folderpath'], 'training_history.csv'),
    model_filepath = os.path.join(config_dict['workspace_folderpath'], 'CVAE_model.hdf5'),
    # parse arguments
    parser = argparse.ArgumentParser(description='Get verbosity.')
    parser.add_argument('--verbose', action='store_true', # default is False
            help='Turn on tqdm progress bar.')
    args = parser.parse_args()
    verbose = args.verbose
    # data
    loader = deeptile_dataset.tile_loader(
            workspace_folderpath=config_dict['workspace_folderpath'],
            warm_start=True,
            image_filepath=config_dict['image_filepath'],
            channel_filepath=config_dict['channel_info_filepath'],
            )
    # model
    cvae_model = deeptile_model.CVAE(
            latent_dim=LATENT_DIM, 
            input_shape=tuple(config_dict['tile_shape'])+(loader.count_channel,),
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            )
    # define grid
    x_linspace = np.linspace(
            start=config_dict['support_range'][0][0], 
            stop=config_dict['support_range'][0][1], 
            num=GRID_NUM,
            ).astype(int)
    y_linspace = np.linspace(
            start=config_dict['support_range'][1][0], 
            stop=config_dict['support_range'][1][1], 
            num=GRID_NUM,
            ).astype(int)
    x_mesh, y_mesh = np.meshgrid(x_linspace, y_linspace)
    survey_grid = []
    for x, y in zip(x_mesh.flatten(), y_mesh.flatten()):
        if deeptile_dataset.within_range(
                point=(x,y),
                tile_shape=config_dict['tile_shape'], 
                support_range=config_dict['support_range']):
            survey_grid.append((x,y))
    survey_grid_array = np.zeros((len(survey_grid), 2))
    for index, (x,y) in enumerate(survey_grid):
        survey_grid_array[index, 0] = x
        survey_grid_array[index, 1] = y
    # convert to dataset
    grid_params = {
            'tile_shape':config_dict['tile_shape'],
            'center_list':survey_grid,
            'batch_size':BATCH_SIZE,
            }
    grid_dataset = loader.get_dataset(**grid_params)
    grid_batch_count = np.ceil(len(survey_grid)/BATCH_SIZE).astype(int)
    # setup training, evaluation, sampling loop
    train_sample = copy.deepcopy(survey_grid)
    record = []
    for cycle in range(TOTAL_CYCLE):
        ts_start = time.time()
        # phase 1: train on sample
        sample_params = {
                'tile_shape':config_dict['tile_shape'],
                'center_list':train_sample,
                'batch_size':BATCH_SIZE,
                }
        sample_dataset = loader.get_dataset(**sample_params)
        sample_batch_count = np.ceil(len(train_sample)/BATCH_SIZE).astype(int)
        train_sample_count = len(train_sample)
        for epoch in range(EPOCH_PER_CYCLE):
            for batch_tile in tqdm.tqdm(
                    iterable=sample_dataset,
                    desc='train',
                    total=sample_batch_count,
                    disable=not verbose):
                cvae_model.compute_apply_gradients(batch_tile)
        # phase 2: survey on grid
        loss_list = []
        for batch_tile in tqdm.tqdm(
                iterable=grid_dataset,
                desc='survey',
                total=grid_batch_count,
                disable=not verbose):
            loss = cvae_model.compute_loss(batch_tile)
            loss_list.append(loss.numpy().flatten())
        prob = np.hstack(loss_list)
        prob /= prob.sum() # note that regression loss >= 0
        # phase 3: generate next training sample
        if verbose:
            print('Multivariate inverse transform sampling...', flush=True)
        train_sample_array = deeptile_sampling.multivariate_inverse_transform_sampling(
            data_X=survey_grid_array,
            data_prob=prob,
            sample_size=len(survey_grid),
            support_range=config_dict['support_range'],
            grid_count=GRID_NUM,
            )
        train_sample = []
        for index in range(train_sample_array.shape[0]):
            if np.isfinite(train_sample_array[index, :]).all()\
                    and deeptile_dataset.within_range(
                            point=train_sample_array[index, :].astype(int),
                            tile_shape=config_dict['tile_shape'], 
                            support_range=config_dict['support_range']):
                        train_sample.append((x, y))
        ts_end = time.time()
        # report progress
        mean_loss = np.mean(np.hstack(loss_list))
        runtime = ts_end-ts_start
        print('cycle {} done, {} samples, grid loss {:.3f}, runtime {:.3f} sec/cycle.'.format(
            cycle, train_sample_count, mean_loss, runtime), flush=True)
        # record history
        record.append([cycle, mean_loss, runtime])
        # check-point
        if not np.isfinite(mean_loss):
            print('non-finite loss detected: {}'.format(mean_loss), flush=True)
            break
    # save record to disk
    df = pd.DataFrame.from_records(record, columns=['cycle', 'loss', 'runtime (sec)'])
    df.to_csv(history_filepath, index=False)
    cvae_model.save(
            filepath=model_filepath,
            overwrite=True,
            include_optimizer=True,
            save_format='h5',
            )
    print('Done.', flush=True)

