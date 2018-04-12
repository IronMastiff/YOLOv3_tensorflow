import tensorflow as tf
import numpy as np
from utils import IOU as get_IOU

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

def calculate_loss( batch_inputs, batch_labels ):
    batch_loss = 0
    # for batch in range( batch_inputs.shape[0] ):
    for image_num in range( batch_inputs.shape[1] ):
        for y in range( batch_inputs.shape[2] ):
            for x in range( batch_inputs.shape[3] ):
                for i in range( 3 ):
                    pretect_x = batch_inputs[image_num][y][x][i * 25]
                    pretect_y = batch_inputs[image_num][y][x][i * 25 + 1]
                    pretect_width = batch_inputs[image_num][y][x][i * 25 + 2]
                    pretect_height = batch_inputs[image_num][y][x][i * 25 + 3]
                    pretect_objectness = batch_inputs[image_num][y][x][i * 25 + 4]
                    pretect_class = batch_inputs[image_num][y][x][i * 25 + 5 : i * 25 + 5 + 20]
                    label_x = batch_labels[image_num][y][x][i * 25]
                    label_y = batch_labels[image_num][y][x][i * 25 + 1]
                    label_width = batch_labels[image_num][y][x][i * 25 + 2]
                    label_height = batch_labels[image_num][y][x][i * 25 + 3]
                    label_objectness = batch_labels[image_num][y][x][i * 25 + 4]
                    label_class = batch_labels[image_num][y][x][i * 25 + 5 : i * 25 + 5 + 20]
                    IOU = get_IOU.IOU_calculator( pretect_x,
                                                  pretect_y,
                                                  pretect_width,
                                                  pretect_height,
                                                  label_x,
                                                  label_y,
                                                  label_width,
                                                  label_height )
                    loss = class_loss( pretect_class,
                                       label_class ) + location_loss( pretect_x,
                                                                      pretect_y,
                                                                      pretect_width,
                                                                      pretect_height,
                                                                      label_x,
                                                                      label_y,
                                                                      label_width,
                                                                      label_height ) + objectness_loss( IOU, label_objectness )
                    batch_loss += loss
    return batch_loss