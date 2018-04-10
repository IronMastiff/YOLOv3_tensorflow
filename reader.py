import tensorflow as tf
import numpy as np
import cv2
from os.path import isfile, join
from os import listdir

def image( batch_size, path ):
    filenames = [join( path, f ) for f in listdir( path ) if isfile( join( path, f ) )]

    batch_filenames = []
    num = len( filenames ) // batch_size
    for i in range( num ):
        batch_filename = filenames[i * batch_size : ( i + 1 ) * batch_size]

        batch_filenames.append( batch_filename )

    if len( filenames ) % batch_size:
        batch_filename = filenames[num * batch_size :]

        batch_filenames.append( batch_filename )

    return batch_filenames

def get_image( path, width, height ):
    image = cv2.imread( path )
    image = cv2.cvtColor( image, cv2.COLOR_BGR2RGB )
    image = cv2.resize( image, ( width, height ) )

    return image



'''--------Test image--------'''
if __name__ == '__main__':
    image = image( 3, './data/VOCtrainval_06-Nov-2007/JPEGImages' )

    print( len( image ) )