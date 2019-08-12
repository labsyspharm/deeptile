import os
import numpy as np
import pandas as pd
import shutil
import typing

import h5py
import tifffile
import tensorflow as tf

class tile_loader(object):
    def __init__(
            self,
            workspace_folderpath: str,
            warm_start: bool=False,
            channel_filepath: str=None,
            image_filepath: str=None,
            ) -> None:
        super().__init__()
        self.workspace_folderpath = workspace_folderpath
        if warm_start:
            # load channel
            channel_filepath = os.path.join(self.workspace_folderpath, 'channel.csv')
            self.channel = pd.read_csv(channel_filepath, index_col=False)
            # load image
            image_filepath = os.path.join(self.workspace_folderpath, 'image.hdf5')
            self.image = h5py.File(image_filepath, 'r')
        else:
            if os.path.isdir(self.workspace_folderpath):
                shutil.rmtree(self.workspace_folderpath)
            os.makedirs(self.workspace_folderpath)
            # process channel information
            self.__parse_channel(channel_filepath)
            # convert image from tiff to hdf5 format for better disk IO performance
            self.__parse_image(image_filepath)
        # useful attribute
        self.count_channel = len(list(self.image.keys()))
        self.image_shape = self.image['channel_0'].shape
        return

    def __parse_channel(self, path: str) -> None:
        with open(path, 'r') as ch:
            channel_name_list = ch.readlines()[0].split(',')
        record = []
        first_hoechst = False
        tile_channel_index = 0
        DNA_name = 'Hoechst'
        background_names = ['A488', 'A555', 'A647']
        for original_channel, channel_name in enumerate(channel_name_list):
            if not first_hoechst and DNA_name in channel_name:
                record.append([original_channel, channel_name, tile_channel_index])
                first_hoechst = True
                tile_channel_index += 1
            elif DNA_name not in channel_name\
                    and not any([bkgd in channel_name for bkgd in background_names]):
                record.append([original_channel, channel_name, tile_channel_index])
                tile_channel_index += 1
            else:
                record.append([original_channel, channel_name, -1])
        self.channel = pd.DataFrame.from_records(record,
                columns=['original_channel', 'channel_name', 'tile_channel'])
        df_path = os.path.join(self.workspace_folderpath, 'channel.csv')
        self.channel.to_csv(df_path, index=False)
        return

    def __parse_image(self, path: str) -> None:
        hdf5_path = os.path.join(self.workspace_folderpath, 'image.hdf5')
        self.image = h5py.File(hdf5_path, 'a')
        original_channel_list = self.channel.loc[self.channel['tile_channel']>=0]\
                .sort_values('tile_channel', ascending=True)\
                ['original_channel'].tolist()
        with tifffile.TiffFile(path) as tif:
            for tile_channel, original_channel in enumerate(original_channel_list):
                wsi = tif.asarray(series=0, key=original_channel)
                wsi = wsi.T # convert from image coordinate to numpy array coordinate
                wsi = wsi.astype(np.float32) # single precision
                wsi -= wsi.mean() # normalization
                wsi /= wsi.std() # same as above
                self.image.create_dataset('channel_{}'.format(tile_channel), data=wsi)
        return

    def __within_image(
            self,
            tile_shape: typing.Tuple[int, int],
            center: typing.Tuple[int, int],
            ) -> bool:
        x_pos, y_pos = center
        x_half_tile_width = tile_shape[0]//2
        y_half_tile_width = tile_shape[1]//2
        image_shape = self.image['channel_0'].shape
        criteria = [
                x_pos-x_half_tile_width >= 0,
                x_pos+x_half_tile_width < image_shape[0],
                y_pos-y_half_tile_width >= 0,
                y_pos+y_half_tile_width < image_shape[1],
                ]
        return all(criteria)

    def get_tile(self, 
            tile_shape: typing.Tuple[int, int],
            center: typing.Tuple[int, int],
            validate: bool=True,
            ) -> np.ndarray:
        if validate and not self.__within_image(tile_shape=tile_shape, center=center):
            return None
        x, y = center
        x_half_width = tile_shape[0]//2
        y_half_width = tile_shape[1]//2
        tile = np.zeros(tile_shape+(self.count_channel,))
        for tile_channel in range(self.count_channel):
            tile_channel_name = 'channel_{}'.format(tile_channel)
            tile[..., tile_channel] = self.image[tile_channel_name][
                    x-x_half_width:x+x_half_width,
                    y-y_half_width:y+y_half_width]
        return tile

    def generate_tiles(self,
            tile_shape: typing.Tuple[int, int],
            center_list: typing.List[typing.Tuple[int, int]],
            ) -> np.ndarray:
        valid_center_list = [c for c in center_list\
                if self.__within_image(tile_shape=tile_shape, center=c)]
        for c in valid_center_list:
            yield self.get_tile(
                    tile_shape=tile_shape,
                    center=c,
                    validate=False,
                    )

    def get_dataset(
            self,
            tile_shape: typing.Tuple[int, int],
            center_list: typing.List[typing.Tuple[int, int]],
            batch_size: int,
            ) -> tf.data.Dataset:
        return tf.data.Dataset.from_generator(
                generator=lambda: self.generate_tiles(
                    tile_shape=tile_shape,
                    center_list=center_list,
                    ),
                output_types=np.float32,
                output_shapes=tile_shape+(self.count_channel,),
                ).batch(batch_size)

