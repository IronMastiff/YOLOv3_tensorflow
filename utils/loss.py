import tensorflow as tf
import numpy as np

def objectness_loss( input, switch, alpha = 0.001 ):
    '''
    Calculate the objectness loss

    :param input: input IOU
    :param switch: Target in this box is 1, else 0
    :return: objectness_loss
    '''

    IOU_loss = tf.square( switch * 0.5 - input )

    if not switch:
        IOU_loss = IOU_loss * alpha

    return IOU_loss

def location_loss( x, y, width, height, l_x, l_y, l_width, l_height, alpha = 0.001 ):
    point_loss = ( tf.square( l_x - x ) + tf.square( l_y - y ) ) * alpha
    size_loss = ( tf.square( tf.sqrt( l_width ) - tf.sqrt( width ) ) + tf.square( tf.sqrt( l_height ) - tf.sqrt( height ) ) ) * alpha

    location_loss = point_loss + size_loss

    return location_loss

def class_loss( input, label ):
    classloss = tf.square( label - input )

    return classloss