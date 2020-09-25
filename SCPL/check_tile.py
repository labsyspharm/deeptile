import os
import sys
import csv

import numpy as np
import pandas as pd
import tqdm

from skimage import io

import celltk

def check_tile(cellid_folderpath, imagerecord_filepath, output_filepath, tile_shape):
    # load data
    image_record = pd.read_csv(imagerecord_filepath)
    image_bounds = {key:(key_id,xspan,yspan) for key_id, key, xspan, yspan in\
            zip(image_record['key_id'].tolist(), image_record['key'].tolist(),
                image_record['x_span'].tolist(), image_record['y_span'].to_list())}

    # write header
    output_file = open(output_filepath, 'w', newline='')
    output_writer = csv.writer(output_file, delimiter=',')
    output_writer.writerow(['key_id', 'cell_id', 'frac_cell_in_tile', 'frac_neighbor_in_tile'])

    # loop over cell ID array
    for cellid_filename in os.listdir(cellid_folderpath):
        # load cell ID array
        key = cellid_filename.split('a0')[0]
        key_id, xbound, ybound = image_bounds[key]
        cellid_filepath = os.path.join(cellid_folderpath, cellid_filename)
        cellid = io.imread(cellid_filepath, memmap=True)
        # loop over cells
        for ci, xi, yi in celltk.iter_cell(cellid):
            # find ranges
            xc, yc = np.median(xi), np.median(yi)
            xl, yl = int(xc-tile_shape[0]/2), int(yc-tile_shape[1]/2)
            xu, yu = xl+tile_shape[0], yl+tile_shape[1]

            # in case cell too close to border to fit the tile
            checkpoint_list = [xl < 0, yl < 0, xu >= xbound, yu >= ybound]
            if any(checkpoint_list):
                continue

            cell_in_tile = (xi >= xl) & (xi < xu) & (yi >= yl) & (yi < yu)
            xi_in_tile, yi_in_tile = xi[cell_in_tile]-xl, yi[cell_in_tile]-yl
            # calculate metrics
            frac_cell_in_tile = cell_in_tile.mean()
            tile = np.ones(tile_shape)
            tile[xi_in_tile, yi_in_tile] = 0
            frac_neighbor_in_tile = tile.mean()
            # logging
            output_writer.writerow([key_id, ci, frac_cell_in_tile, frac_neighbor_in_tile])

    # close file
    output_file.close()

if __name__ == '__main__':
    # input path
    data_folderpath = '/n/scratch2/hungyiwu/project_deeptile/data'
    cellid_folderpath = os.path.join(data_folderpath, 'Basel_Zuri_masks')
    imagerecord_filepath = os.path.join(data_folderpath, 'image_record.csv')
#    tileshape_list = [(i,i) for i in range(5, 32, 2)]
    tileshape_list = [(45, 45)]
    job_id = int(sys.argv[1])

    # run
    tile_shape = tileshape_list[job_id]
    output_filepath = os.path.join(data_folderpath, 'checktile_{}x{}.csv'.format(*tile_shape))
    check_tile(cellid_folderpath, imagerecord_filepath, output_filepath, tile_shape)
