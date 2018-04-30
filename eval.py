import time
import tensorflow as tf
import os
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np

import reader
from utils import net, read_config, select_things, eval_uitls

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( '-c', '--conf', default = './config/eval_config.yml', help = 'the path to the eval_config file' )
    return parser.parse_args()

def main( FLAGS ):
    if not os.path.exists( FLAGS.save_dir ):
        os.makedirs( FLAGS.save_dir )

    input_image = reader.get_image( FLAGS.image_dir, FLAGS.image_width, FLAGS.image_height )
    output_image = np.copy( input_image )

    '''--------Create placeholder--------'''
    image = net.create_eval_placeholder( FLAGS.image_width, FLAGS.image_height )

    '''--------net--------'''
    pre_scale1, pre_scale2, pre_scale3 = net.feature_extractor( image, False )
    scale1, scale2, scale3 = net.scales( pre_scale1, pre_scale2, pre_scale3, False )

    with tf.Session() as sess:
        saver = tf.train.Saver()
        save_path = select_things.select_checkpoint( FLAGS.scale )
        last_checkpoint = tf.train.latest_checkpoint( save_path, 'checkpoint' )
        if last_checkpoint:
            saver.restore(sess, last_checkpoint)
            print( 'Success load model from: ', format( last_checkpoint ) )
        else:
            print( 'Model has not trained' )

        start_time = time.time()
        scale1, scale2, scale3 = sess.run( [scale1, scale2, scale3], feed_dict = {image: [output_image]} )

    if FLAGS.scale == 1:
        scale = scale1
    if FLAGS.scale == 2:
        scale = scale2
    if FLAGS.scale == 3:
        scale = scale3

    boxes_labels = eval_uitls.label_extractor( scale[0] )

    bdboxes = eval_uitls.get_bdboxes( boxes_labels )

    for bdbox in bdboxes:
        font = cv2.FONT_HERSHEY_SIMPLEX
        output_image = cv2.rectangle( output_image,
                                      ( int( bdbox[0] - bdbox[2] / 2 ), int( bdbox[1] - bdbox[3] / 2 ) ),
                                      ( int( bdbox[0] + bdbox[2] / 2 ), int( bdbox[1] + bdbox[3] / 2 ) ),
                                      ( 200, 0, 0 ),
                                      1 )
        output_image = cv2.putText( output_image,
                                    bdbox[4],
                                    ( int( bdbox[0] - bdbox[2] / 2 ), int( bdbox[1] - bdbox[3] / 2 ) ),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.3,
                                    (0, 255, 0),
                                    1 )
    # output_image = np.multiply( output_image, 255 )

    generate_image = FLAGS.save_dir + '/res.jpg'
    if not os.path.exists( FLAGS.save_dir ):
        os.makedirs( FLAGS.save_dir )

    cv2.imwrite( generate_image, cv2.cvtColor( output_image, cv2.COLOR_RGB2BGR ) )
    end_time = time.time()

    print( 'Use time: ', end_time - start_time )

    plt.imshow( output_image )
    plt.show()




if __name__ == '__main__':
    args = parse_args()
    FLAGS = read_config.read_config_file( args.conf )
    main( FLAGS )