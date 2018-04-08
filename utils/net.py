import tensorflow as tf
import munpy as np

def create_placeholder( inputs, labels ):
    X = tf.placeholder( tf.float32, [None, input.shape[1], input.shape[2], 3] )
    Y = tf.placeholder( tf.float32, [None, labels.shape[1], labels.shape[2], 3] )

    return X, Y


def Leaky_Relu( input, alpha = 0.01 ):
    output = tf.maximum( input, tf.multiply( input, alpha ) )

    return output


def conv2d( inputs, filters, shape, stried = ( 1, 1 ) ):
    layer = tf.layers.conv2d( inputs,
                              filters,
                              shape,
                              stride,
                              padding = 'SAME',
                              kernel_initializer = tf.truncated_normal_initializer( stddev = 0.01 ) )

    layer = tf.layers.batch_normalization( layer, training = True )

    layer = Leaky_Relu( layer )

    return layer


def Res_conv2d( inputs, filter, shape, stried = ( 1, 1 ) ):
    layer = tf.layers.conv2d( inputs,
                              shape,
                              stried,
                              padding = 'SAME',
                              kernel_initializer = tf.truncated_normal_initializer( stddec = 0.01 ) )

    layer = tf.layers.batch_normalization( layer, training = True )

    return layer

def Res( input, shortcut ):

    layer = Leaky_Relu( tf.add( input, shortcut ) )

    return layer




def net( inputs ):
    layer = conv2d( inputs, 32, [3, 3] )
    layer = conv2d( layer, 64, [3, 3], ( 2, 2 ) )

    layer = conv2d( layer, 32, [1, 1] )
    shortcut = layer
    layer = Res_conv2d( layer, 64, [3, 3] )
    layer = Res( layer, shortcut )

    layer = conv2d( layer, 128, [3, 3], ( 2, 2 ) )

    for _ in range( 2 ):
        layer = conv2d( layer, 64, [1, 1] )
        shortcut = layer
        layer = Res_conv2d( layer, 128, [3, 3] )
        layer = Res( layer, shortcut )

    layer = conv2d( layer, 256, [3, 3], ( 2, 2 ) )

    for _ in range( 8 ):
        layer = conv2d( layer, 128, [1, 1] )
        shortcut = layer
        layer = Res_conv2d( layer, 256, [3, 3] )
        layer = Res( layer, shortcut )

    layer = covn2d( layer, 512, [3, 3], ( 2, 2 ) )

    for _ in range( 8 ):
        layer = conv2d( layer, 256, [1, 1] )
        shortcut = layer
        layer = Res_conv2d( layer, 512, [3, 3] )
        layer = Res( layer, shortcut )

    layer = conv2d( layer, 1024, [3, 3], ( 2, 2 ) )

    for _ in range( 4 ):
        layer = con2d( layer, 512, [1, 1] )
        shortcut =layer
        layer = Res_conv2d( layer, 1024, [3, 3] )
        layer = Res( layer, shortcut )

    avg_pool = tf.nn.pool( layer, 1, 'AVG', 'SAME' )
    softmax = tf.nn.softmax( avg_pool, )

    return layer, avg_pool, softmax
