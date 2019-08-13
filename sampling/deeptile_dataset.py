import os
import numpy as np
import pandas as pd
import shutil
import typing

import h5py
import tifffile
import tensorflow as tf

def coordinate_convert(
        point: typing.Tuple[int, int],
        shape: typing.Tuple[int, int],
        from_coordinate: str,
        to_coordinate: str,
        ) -> typing.Tuple[int, int]:
    '''
    Image coordinate:
    * x-axis is horizontal, ascending to the right
    * y-axis is vertical, ascending to the bottom

    Plot coordinate:
    * x-axis is horizontal, ascending to the right
    * y-axis is vertical, ascending to the top

    Array coordinate:
    * x (row) is vertical, ascending to the bottom
    * y (column) is horizontal, ascending to the right
    '''
    # convert input to array coordinate as standard
    if from_coordinate == 'image':
        x = point[1]
        y = point[0]
    elif from_coordinate == 'plot':
        x = shape[1]-point[1]
        y = point[0]
    elif from_coordinate == 'array':
        x, y = point
    else:
        raise ValueError('Argument from_coordinate {} not recognized.'\
                'Accepted values are {{image, plot, array}}.'.format(from_coordinate))
    # convert array coordinate to output coordinate
    if to_coordinate == 'image':
        new_x = y
        new_y = x
    elif to_coordinate == 'plot':
        new_x = y
        new_y = shape[0]-x
    elif to_coordinate == 'array':
        new_x = x
        new_y = y
    else:
        raise ValueError('Argument to_coordinate {} not recognized.'\
                'Accepted values are {{image, plot, array}}.'.format(to_coordinate))
    return (new_x, new_y)

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
        return

    @property
    def count_channel(self):
        return (self.channel['tile_channel'] >= 0).sum()

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
        # for consistent preprocessing
        def preprocessing(a: np.ndarray) -> np.ndarray:
            a = a.astype(np.float32) # single precision
            a -= a.mean() # normalization
            a /= a.std() # same as above
            return a
        # get channel list
        original_channel_list = self.channel.loc[self.channel['tile_channel']>=0]\
                .sort_values('tile_channel', ascending=True)\
                ['original_channel'].tolist()
        with tifffile.TiffFile(path) as tif:
            # initialization
            tile_channel = 0
            original_channel = original_channel_list[tile_channel]
            wsi = tif.asarray(series=0, key=original_channel)
            wsi = preprocessing(wsi)
            image_shape = wsi.shape+(len(original_channel_list),)
            hdf5_path = os.path.join(self.workspace_folderpath, 'image.hdf5')
            self.image = h5py.File(hdf5_path, 'a')
            self.image.create_dataset('image', shape=image_shape, dtype=np.float32, chunks=True)
            # fill first channel
            self.image['image'][..., tile_channel] = wsi
            tile_channel += 1
            # fill rest of the channels
            for original_channel in original_channel_list[1:]:
                wsi = tif.asarray(series=0, key=original_channel)
                wsi = preprocessing(wsi)
                self.image['image'][..., tile_channel] = wsi
                tile_channel += 1
        return

    def __within_image(
            self,
            tile_shape: typing.Tuple[int, int],
            center: typing.Tuple[int, int],
            ) -> bool:
        '''
        Input is in image coordinate; data is in array coordinate.
        '''
        # convert all input to array coordinate
        wsi_array_shape = self.image['image'].shape
        wsi_image_shape = (wsi_array_shape[1], wsi_array_shape[0])
        x, y = coordinate_convert(
                point=center,
                shape=wsi_image_shape,
                from_coordinate='image',
                to_coordinate='array',
                )
        tile_array_shape = (tile_shape[1], tile_shape[0])
        # now check criteria
        x_half_tile_width = int(tile_array_shape[0]/2)
        y_half_tile_width = int(tile_array_shape[1]/2)
        criteria = [
                x-x_half_tile_width >= 0,
                x+x_half_tile_width < wsi_array_shape[0],
                y-y_half_tile_width >= 0,
                y+y_half_tile_width < wsi_array_shape[1],
                ]
        return all(criteria)

    def get_tile(self, 
            tile_shape: typing.Tuple[int, int],
            center: typing.Tuple[int, int],
            validate: bool=True,
            ) -> np.ndarray:
        '''
        Input is in image coordinate; data and output are in array coordinate.
        '''
        # self.__within_image takes image coordinate as input
        if validate and not self.__within_image(tile_shape=tile_shape, center=center):
            return None
        # convert all input to array coordinate
        wsi_array_shape = self.image['image'].shape
        wsi_image_shape = (wsi_array_shape[1], wsi_array_shape[0])
        x, y = coordinate_convert(
                point=center,
                shape=wsi_image_shape,
                from_coordinate='image',
                to_coordinate='array',
                )
        tile_array_shape = (tile_shape[1], tile_shape[0])
        # now get tile
        x_half_width = int(tile_array_shape[0]/2)
        y_half_width = int(tile_array_shape[1]/2)
        return self.image['image'][
                    x-x_half_width:x+x_half_width,
                    y-y_half_width:y+y_half_width, :]

    def generate_tiles(self,
            tile_shape: typing.Tuple[int, int],
            center_list: typing.List[typing.Tuple[int, int]],
            ) -> np.ndarray:
        valid_center_list = self.within_image(
                tile_shape=tile_shape,
                center_list=center_list,
                )
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
        tile_array_shape = (tile_shape[1], tile_shape[0])
        return tf.data.Dataset.from_generator(
                generator=lambda: self.generate_tiles(
                    tile_shape=tile_shape,
                    center_list=center_list,
                    ),
                output_types=np.float32,
                output_shapes=tile_array_shape+(self.count_channel,),
                ).batch(batch_size)

    def within_image(
            self,
            tile_shape: typing.Tuple[int, int],
            center_list: typing.List[typing.Tuple[int, int]],
            ) -> int:
        return [c for c in center_list\
                if self.__within_image(tile_shape=tile_shape, center=c)]
