import cv2
import tensorflow as tf
import skimage.transform


def create_placeholder( batch_size, width, height, final_width, final_height ):
    X = tf.placeholder( tf.float32, [batch_size, width, height, 3] )
    Y = tf.placeholder( tf.float32, [batch_size, final_width, final_height, 255] )
    train = tf.placeholder( tf.bool )

    return X, Y, train

def create_eval_placeholder( width, height ):
    image = tf.placeholder( tf.float32, [1, width, height, 3] )

    return image

def Leaky_Relu( input, alpha = 0.01 ):
    output = tf.maximum( input, tf.multiply( input, alpha ) )

    return output


def conv2d( inputs, filters, shape, stride = ( 1, 1 ), training = True ):
    layer = tf.layers.conv2d( inputs,
                              filters,
                              shape,
                              stride,
                              padding = 'SAME',
                              kernel_initializer=tf.truncated_normal_initializer( stddev=0.01 ) )

    layer = tf.layers.batch_normalization( layer, training = training )

    layer = Leaky_Relu( layer )

    return layer


def Res_conv2d( inputs, shortcut, filters, shape, stride = ( 1, 1 ), training = True ):
    conv = conv2d( inputs, filters, shape, training = training )
    Res = Leaky_Relu( conv + shortcut )

    return Res


def feature_extractor( inputs, training ):
    layer = conv2d( inputs, 32, [3, 3], training = training )
    layer = conv2d( layer, 64, [3, 3], ( 2, 2 ), training = training )
    shortcut = layer

    layer = conv2d( layer, 32, [1, 1], training = training )
    layer = Res_conv2d( layer, shortcut, 64, [3, 3], training = training )

    layer = conv2d( layer, 128, [3, 3], ( 2, 2 ), training = training )
    shortcut = layer

    for _ in range( 2 ):
        layer = conv2d( layer, 64, [1, 1], training = training )
        layer = Res_conv2d( layer, shortcut, 128, [3, 3], training = training )

    layer = conv2d( layer, 256, [3, 3], ( 2, 2 ), training = training )
    shortcut = layer

    for _ in range( 8 ):
        layer = conv2d( layer, 128, [1, 1], training = training )
        layer = Res_conv2d( layer, shortcut, 256, [3, 3], training = training )
    pre_scale3 = layer

    layer = conv2d( layer, 512, [3, 3], ( 2, 2 ), training = training )
    shortcut = layer

    for _ in range( 8 ):
        layer = conv2d( layer, 256, [1, 1], training = training )
        layer = Res_conv2d( layer, shortcut, 512, [3, 3], training = training )
    pre_scale2 = layer

    layer = conv2d( layer, 1024, [3, 3], ( 2, 2 ), training = training )
    shortcut = layer

    for _ in range( 4 ):
        layer = conv2d( layer, 512, [1, 1], training = training )
        layer = Res_conv2d( layer, shortcut, 1024, [3, 3], training = training )
    pre_scale1 = layer

    return pre_scale1, pre_scale2, pre_scale3

def get_layer2x( layer_final, pre_scale ):
    layer2x = tf.image.resize_images(layer_final,
                                     [2 * tf.shape(layer_final)[1], 2 * tf.shape(layer_final)[2]])
    layer2x_add = tf.concat( [layer2x, pre_scale], 3 )

    return layer2x_add

def scales( layer, pre_scale2, pre_scale3, training ):
    layer_copy = layer
    layer = conv2d( layer, 512, [1, 1], training = training )
    layer = conv2d( layer, 1024, [3, 3], training = training )
    layer = conv2d(layer, 512, [1, 1], training = training )
    layer_final = layer
    layer = conv2d(layer, 1024, [3, 3], training = training )

    '''--------scale_1--------'''
    scale_1 = conv2d( layer, 255, [1, 1], training = training )

    '''--------scale_2--------'''
    layer = conv2d( layer_final, 256, [1, 1], training = training )
    layer = get_layer2x( layer, pre_scale2 )

    layer = conv2d( layer, 256, [1, 1], training = training )
    layer= conv2d( layer, 512, [3, 3], training = training )
    layer = conv2d( layer, 256, [1, 1], training = training )
    layer = conv2d( layer, 512, [3, 3], training = training )
    layer = conv2d( layer, 256, [1, 1], training = training )
    layer_final = layer
    layer = conv2d( layer, 512, [3, 3], training = training )
    scale_2 = conv2d( layer, 255, [1, 1], training = training )

    '''--------scale_3--------'''
    layer = conv2d( layer_final, 128, [1, 1], training = training )
    layer = get_layer2x( layer, pre_scale3 )

    for _ in range( 3 ):
        layer = conv2d( layer, 128, [1, 1], training = training )
        layer = conv2d( layer, 256, [3, 3], training = training )
    scale_3 = conv2d( layer, 255, [1, 1], training = training )

    scale_1 = tf.abs( scale_1 )
    scale_2 = tf.abs( scale_2 )
    scale_3 = tf.abs( scale_3 )

    return scale_1, scale_2, scale_3







'''--------Test the scale--------'''
if __name__ == "__main__":
    data = cv2.imread(  '../data/VOCtest_06-Nov-2007/JPEGImages/000001.jpg' )
    data = cv2.cvtColor( data, cv2.COLOR_BGR2RGB )
    data = cv2.resize( data, ( 416, 416 ) )

    data = tf.cast( tf.expand_dims( tf.constant( data ), 0 ), tf.float32 )

    pre_scale1, pre_scale2, pre_scale3 = feature_extractor( data )

    scale_1, scale_2, scale_3 = scales( pre_scale1, pre_scale2, pre_scale3 )

    with tf.Session() as sess:

        sess.run( tf.initialize_all_variables() )

        print( sess.run( scale_1 ).shape )