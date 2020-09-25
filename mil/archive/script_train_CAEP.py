import numpy as np
import pandas as pd
import time
import argparse
import os
import copy
import yaml

import tqdm
import tensorflow as tf

import CAEP
import preprocessing

if __name__ == '__main__':
    # parameter defined within the scope of model preparation
    BATCH_SIZE = 10
    LATENT_DIM = 20
    LEARNING_RATE = 1e-5
    GRID_NUM = 1000
    TOTAL_EPOCH = 10
    # load configuration
    master_folderpath = '/n/scratch2/hungyiwu/deeptile_data/'
    triplet_folderpath = '/n/scratch2/hungyiwu/triplet_images'
    config_filepath = os.path.expanduser('./default_config.yaml')
    with open(config_filepath, 'r') as yaml_file:
        config_dict = yaml.safe_load(yaml_file)
    # paths
    model_filepath = os.path.join(master_folderpath, 'CAEP_model.hdf5'),
    # parse arguments
    parser = argparse.ArgumentParser(description='Get verbosity.')
    parser.add_argument('--verbose', action='store_true', # default is False
            help='Turn on tqdm progress bar.')
    args = parser.parse_args()
    verbose = args.verbose
    # data
    patient_id = '26531'
    pre_identifier = patient_id+'PRE'
    pre_loader = preprocessing.tile_loader(
            workspace_folderpath=os.path.join(master_folderpath, pre_identifier),
            warm_start=True,
            image_filepath=os.path.join(triplet_folderpath, pre_identifier+'.ome.tif'),
            channel_filepath=os.path.join(master_folderpath, patient_id+'_channel_info.csv'),
            )
    post_identifier = patient_id+'PRE'
    post_loader = preprocessing.tile_loader(
            workspace_folderpath=os.path.join(master_folderpath, post_identifier),
            warm_start=True,
            image_filepath=os.path.join(triplet_folderpath, post_identifier+'.ome.tif'),
            channel_filepath=os.path.join(master_folderpath, patient_id+'_channel_info.csv'),
            )
    # model
    caep_model = CAEP.CAEP(
            latent_dim=LATENT_DIM, 
            feature_shape=tuple(config_dict['tile_shape'])+(pre_loader.count_channel,),
            )
    caep_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    # define grid
    def get_grid(image_shape, grid_num):
        x_linspace = np.linspace(
                start=0, 
                stop=image_shape[0], 
                num=GRID_NUM,
                ).astype(int)
        y_linspace = np.linspace(
                start=0, 
                stop=image_shape[1], 
                num=GRID_NUM,
                ).astype(int)
        x_mesh, y_mesh = np.meshgrid(x_linspace, y_linspace)
        survey_grid = []
        for x, y in zip(x_mesh.flatten(), y_mesh.flatten()):
            if preprocessing.within_range(
                    point=(x,y),
                    tile_shape=config_dict['tile_shape'], 
                    support_range=[(0, image_shape[0]), (0, image_shape[1])],
                    ):
                survey_grid.append((x,y))
        return survey_grid
    pre_grid = get_grid(pre_loader.image['image'].shape, GRID_NUM)
    post_grid = get_grid(post_loader.image['image'].shape, GRID_NUM)
    # convert to dataset
    pre_dataset = pre_loader.get_dataset(
        tile_shape=config_dict['tile_shape'],
        center_list=pre_grid,
        batch_size=BATCH_SIZE,
        )
    pre_batch_count = np.ceil(len(pre_grid)/BATCH_SIZE).astype(int)
    post_dataset = post_loader.get_dataset(
        tile_shape=config_dict['tile_shape'],
        center_list=post_grid,
        batch_size=BATCH_SIZE,
        )
    post_batch_count = np.ceil(len(post_grid)/BATCH_SIZE).astype(int)
    # setup training, evaluation, sampling loop
    record = []
    for epoch in range(TOTAL_EPOCH):
        ts_start = time.time()
        # phase 1: train on grid
        for pre_batch, post_batch in tqdm.tqdm(
                iterable=zip(pre_dataset, post_dataset),
                desc='train',
                total=min(pre_batch_count, post_batch_count),
                disable=not verbose):
            caep_model.compute_apply_gradients(pre_batch, post_batch, caep_optimizer)
        # phase 2: survey on grid
        loss_list = []
        for pre_batch, post_batch in tqdm.tqdm(
                iterable=zip(pre_dataset, post_dataset),
                desc='evaluate',
                total=min(pre_batch_count, post_batch_count),
                disable=not verbose):
            loss = caep_model.compute_loss(pre_batch, post_batch)
            loss_list.append(loss.numpy()[0])
        ts_end = time.time()
        # report progress
        mean_loss = np.mean(loss_list)
        runtime = ts_end-ts_start
        print('epoch {}, loss {:.3f}, runtime {:.3f} sec/epoch.'.format(
            epoch, mean_loss, runtime), flush=True)
        # check-point
        if not np.isfinite(mean_loss):
            print('non-finite loss detected: {}'.format(mean_loss), flush=True)
            break
    caep_model.save(
            filepath=model_filepath,
            overwrite=True,
            include_optimizer=True,
            save_format='h5',
            )
    print('Done.', flush=True)

