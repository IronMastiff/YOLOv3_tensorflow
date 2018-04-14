import numpy as np
import os


def select_scale(tag, image_width, image_height):
    if tag == 1:
        scale_width = image_width / np.power( 2, 5 )
        scale_height = image_height / np.power( 2, 5 )
    if tag == 2:
        scale_width = image_width / np.power( 2, 4 )
        scale_height = image_height / np.power( 2, 4 )
    if tag == 3:
        scale_width = image_width / np.power( 2, 3 )
        scale_height = image_height / np.power( 2, 3 )

    return scale_width, scale_height


def select_checkpoint( tag ):
    dir_name = 'scale' + str( tag )

    checkpoint_path = os.path.join( './models', dir_name )

    if not (os.path.exists( checkpoint_path ) ):
        os.makedirs( checkpoint_path )

    return checkpoint_path