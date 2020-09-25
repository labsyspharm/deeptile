import os
import csv
import shutil

import numpy as np
import tqdm

from skimage import io

import celltk

if __name__ == '__main__':
    # input path
    data_folderpath = '/n/scratch2/hungyiwu/project_deeptile/data'
    image_folderpath = os.path.join(data_folderpath, 'tif_images')
    cellid_folderpath = os.path.join(data_folderpath, 'Basel_Zuri_masks')
    marker_filepath = os.path.join(data_folderpath, 'markers.csv')

    # output path
    output_folderpath = os.path.join(data_folderpath, 'scaled_images')
    cellrecord_filepath = os.path.join(data_folderpath, 'cell_record.csv')
    imagerecord_filepath = os.path.join(data_folderpath, 'image_record.csv')

    # load data
    with open(marker_filepath, 'r', newline='') as infile:
        reader = csv.reader(infile, delimiter=',')
        channel_index = [row[0] for row in reader]
        channel_index = [int(s) for s in channel_index[1:]] # exclude header

    if os.path.isdir(output_folderpath):
        shutil.rmtree(output_folderpath)
    os.mkdir(output_folderpath)

    # file matching
    image_key_list = [n.split('a0')[0] for n in os.listdir(image_folderpath)]
    cellid_key_list = [n.split('a0')[0] for n in os.listdir(cellid_folderpath)]
    key_set = set(image_key_list + cellid_key_list)
    get_key_id_dict = {key:i for i, key in enumerate(key_set)}
    path_dict = {}
    for image_filename in os.listdir(image_folderpath):
        key = image_filename.split('a0')[0]
        if key in key_set:
            path_dict[key] = {'image_filename': image_filename,
                    'key_id': get_key_id_dict[key]}
    for cellid_filename in os.listdir(cellid_folderpath):
        key = cellid_filename.split('a0')[0]
        if key in key_set:
            path_dict[key]['cellid_filename'] = cellid_filename

    # write record
    cellrecord_file = open(cellrecord_filepath, 'w', newline='')
    cellrecord_writer = csv.writer(cellrecord_file, delimiter=',')
    cellrecord_writer.writerow(['key_id', 'cell_id', 'x_start', 'x_end', 'x_median',
        'y_start', 'y_end', 'y_median', 'area'])

    imagerecord_file = open(imagerecord_filepath, 'w', newline='')
    imagerecord_writer = csv.writer(imagerecord_file, delimiter=',')
    imagerecord_writer.writerow(['key_id', 'key', 'image_filename', 'cellid_filename',
        'x_span', 'y_span'])

    # loop over images
    for key in tqdm.tqdm(path_dict):
        # load images
        image_filepath = os.path.join(image_folderpath, path_dict[key]['image_filename'])
        cellid_filepath = os.path.join(cellid_folderpath, path_dict[key]['cellid_filename'])
        image = io.imread(image_filepath, memmap=True)
        cellid = io.imread(cellid_filepath, memmap=True)
        # record image metadata
        key_id = get_key_id_dict[key]
        imagerecord_writer.writerow([key_id, key, path_dict[key]['image_filename'],
            path_dict[key]['cellid_filename'], image.shape[1], image.shape[2]])
        # normalize image and save to disk
        scaled_image = np.zeros((image.shape[1], image.shape[2], len(channel_index)))
        for new_channel, old_channel in enumerate(channel_index):
            try:
                scaled_image[..., new_channel] = celltk.percentile_scale(
                        image[old_channel, ...])
            except np.linalg.LinAlgError:
                print('percentile scaling failed for key {} channel {}'.format(
                    key, old_channel), flush=True)
                continue
        output_filepath = os.path.join(output_folderpath, key+'.npy')
        np.save(output_filepath, scaled_image)
        # loop over cells
        for ci, xi, yi in celltk.iter_cell(cellid):
            # record cell metadata
            cellrecord_writer.writerow([key_id, ci, xi.min(), xi.max(), np.median(xi),
                yi.min(), yi.max(), np.median(yi), xi.shape[0]])

    # close files
    cellrecord_file.close()
    imagerecord_file.close()
    print('all done.', flush=True)
