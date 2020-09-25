import os

import preprocessing

if __name__ == '__main__':
    master_folderpath = '/n/scratch2/hungyiwu/deeptile_data/'
    image_folder = '/n/scratch2/hungyiwu/triplet_images/'
    image_name_list = [n for n in os.listdir(image_folder) if n.endswith('.ome.tif')]
    for image_name in image_name_list:
        image_identifier = image_name.rstrip('.tif').rstrip('.ome')
        workspace_folderpath = os.path.join(master_folderpath, image_identifier)
        loader = preprocessing.tile_loader(
                workspace_folderpath=workspace_folderpath,
                warm_start=False,
                channel_filepath=os.path.join(master_folderpath, image_identifier+'_channel_info.csv'),
                image_filepath=os.path.join(image_folder, image_name),
                )

