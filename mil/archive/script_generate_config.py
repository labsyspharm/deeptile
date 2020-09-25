import yaml
import os

if __name__ == '__main__':
    # define tile shape
    # note: shape is based on image coordinate
    displacement_cells = 10 # width/2, in unit of cells
    cell_size = 10 # in unit of micro-meter
    pixel_size = 0.65 # in unit of micro-meter
    tile_width = 2*int(displacement_cells * cell_size / pixel_size) # in unit of pixel
    tile_shape = (tile_width, tile_width)
    # dump yaml
    output_dict = {
            'tile_shape':tile_shape,
            }
    config_filepath = os.path.expanduser('./default_config.yaml')
    if os.path.isfile(config_filepath):
        os.remove(config_filepath)
    with open(config_filepath, 'w') as yaml_file:
        yaml.safe_dump(output_dict, stream=yaml_file)
    print('Done.')
