import numpy as np
import tensorflow as tf

def calculate_min( point, data ):
    min_point = point - data

    return min_point

def calculate_max( point, data ):
    max_point = point + data

    return max_point

def IOU_calculator( x, y, width, height, l_x, l_y, l_width, l_height ):
    '''
    Cculate IOU

    :param x: net predicted x
    :param y: net predicted y
    :param width: net predicted width
    :param height: net predicted height
    :param l_x: label x
    :param l_y: label y
    :param l_width: label width
    :param l_height: label height
    :return: IOU
    '''

    x_max = calculate_max( x , width / 2 )
    y_max = calculate_max( y, height / 2 )
    x_min = calculate_min( x, width / 2 )
    y_min = calculate_min( y, height / 2 )

    l_x_max = calculate_max( l_x, width / 2 )
    l_y_max = calculate_max( l_y, height / 2 )
    l_x_min = calculate_min( l_x, width / 2 )
    l_y_min = calculate_min( l_y, height / 2 )

    '''--------Caculate Both Area's point--------'''
    xend = tf.minimum( x_max, l_x_max )
    xstart = tf.maximum( x_min, l_x_min )

    yend = tf.minimum( y_max, l_y_max )
    ystart = tf.maximum( y_min, l_y_min )

    area_width = xend - xstart
    area_height = yend - ystart

    '''--------Caculate the IOU--------'''
    area = area_width * area_height

    all_area = tf.cond( ( width * height + l_width * l_height - area ) <= 0, lambda : tf.cast( 1e-8, tf.float32 ), lambda : ( width * height + l_width * l_height - area ) )

    IOU = area / all_area

    IOU = tf.cond( area_width < 0, lambda : tf.cast( 1e-8, tf.float32 ), lambda : IOU )
    IOU = tf.cond( area_height < 0, lambda : tf.cast( 1e-8, tf.float32 ), lambda : IOU )

    return IOU



'''--------Test the IOU function--------'''
if __name__ == '__main__':
    IOU1 = IOU_calculator( tf.cast( 1, tf.float32 ), tf.cast( 1, tf.float32 ), tf.cast( 2, tf.float32 ), tf.cast( 2, tf.float32 ),
                          tf.cast( 2, tf.float32 ), tf.cast( 2, tf.float32 ), tf.cast( 2, tf.float32 ), tf.cast( 2, tf.float32 ) )
    IOU = IOU_calculator(tf.cast( 0, tf.float32 ), tf.cast( 0, tf.float32 ), tf.cast( 0, tf.float32 ), tf.cast( 0, tf.float32 ),
                         tf.cast( 0, tf.float32 ), tf.cast( 0, tf.float32 ), tf.cast( 0, tf.float32 ), tf.cast( 0, tf.float32 ) )
    sess = tf.Session()
    print( sess.run( IOU ) )