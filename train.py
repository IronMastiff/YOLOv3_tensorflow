import tensorflow as tf
import numpy as np
import os
import argparse
import time
import utils.read_config as read_config

from utils import net, read_config, get_loss, IOU, extract_labels, select_things
import reader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( '-c', '--conf', default = './config/config.yml', help = 'the path to the config file' )
    return parser.parse_args()

def main( FLAGS ):

    scale_width, scale_height = select_things.select_scale( FLAGS.scale, FLAGS.width, FLAGS.height )
    '''--------Creat palceholder--------'''
    datas, labels, train = net.create_placeholder( FLAGS.batch_size, FLAGS.width, FLAGS.height, scale_width, scale_height )

    '''--------net--------'''
    pre_scale1, pre_scale2, pre_scale3 = net.feature_extractor( datas, train )
    scale1, scale2, scale3 = net.scales( pre_scale1, pre_scale2, pre_scale3, train )

    '''--------get labels_filenames and datas_filenames--------'''
    datas_filenames = reader.images( FLAGS.batch_size, FLAGS.datas_path )
    labels_fienames = reader.labels( FLAGS.batch_size, FLAGS.labels_path )
    normalize_labels = extract_labels.labels_normalizer( labels_fienames,
                                                          FLAGS.width,
                                                          FLAGS.height,
                                                          scale_width,
                                                          scale_height )

    '''---------partition the train data and val data--------'''
    train_filenames = datas_filenames[: int( len( datas_filenames ) * 0.9 )]
    train_labels = normalize_labels[: int( len( normalize_labels ) * 0.9 )]
    val_filenames = datas_filenames[len( datas_filenames ) - int( len( datas_filenames ) * 0.9 ) :]
    val_labels = normalize_labels[len( normalize_labels ) - int( len( normalize_labels ) * 0.9 ) :]

    '''--------calculate loss--------'''
    if FLAGS.scale == 1:
        loss = get_loss.calculate_loss( scale1, labels )

    if FLAGS.scale == 2:
        loss = get_loss.calculate_loss( scale2, labels )

    if FLAGS.scale == 3:
        loss = get_loss.calculate_loss( scale3, labels )

    '''--------Optimizer--------'''
    update_ops = tf.get_collection( tf.GraphKeys.UPDATE_OPS )
    with tf.control_dependencies( update_ops ):
        optimizer = tf.train.AdamOptimizer( learning_rate=FLAGS.learning_rate ).minimize( loss )

    tf.summary.scalar( 'loss',  loss )
    merged = tf.summary.merge_all()

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter( "logs/", sess.graph )
        number = 0

        saver = tf.train.Saver( max_to_keep = 10 )
        save_path = select_things.select_checkpoint( FLAGS.scale )
        last_checkpoint = tf.train.latest_checkpoint( save_path, 'checkpoint' )
        if last_checkpoint:
            saver.restore( sess, last_checkpoint )
            number = int( last_checkpoint[28 :] ) + 1
            print( 'Reuse model form: ', format( last_checkpoint ) )
        else:
            sess.run( init )


        for epoch in range( FLAGS.epoch ):
            epoch_loss = tf.cast( 0, tf.float32 )
            for i in range( len( train_filenames ) ):
                normalize_datas = []
                for data_filename in train_filenames[i]:
                    image = reader.get_image( data_filename, FLAGS.width, FLAGS.height )
                    image = np.array( image, np.float32 )

                    normalize_datas.append( image )

                normalize_datas = np.array( normalize_datas )

                _, batch_loss, rs = sess.run( [optimizer, loss, merged], feed_dict = {datas: normalize_datas, labels: train_labels[i], train: True} )

                epoch_loss =+ batch_loss

            writer.add_summary( rs, epoch + number )


            if epoch % 1 == 0 & epoch != 0:
                print( 'Cost after epoch %i: %f' % ( epoch + number, epoch_loss ) )
                name = 'scale' + str( FLAGS.scale ) + '.ckpt'
                saver.save( sess, os.path.join( save_path, name ), global_step = epoch + number )

            if epoch % 10 == 0 & epoch != 0:
                val_loss = tf.cast( 0, tf.float32 )
                for i in range( len( val_filenames ) ):
                    normalize_datas = []
                    for val_filename in val_filenames[i]:
                        image = reader.get_image( val_filename, FLAGS.width, FLAGS.height )
                        image = np.array( image, np.float32 )
                        image = np.divide( image, 255 )

                        normalize_datas.append( image )

                    normalize_datas = np.array( normalize_datas )

                    batch_loss = sess.run( loss, feed_dict = {datas: normalize_datas, labels: val_labels[i], train: False} )

                    val_loss =+ batch_loss

                print( 'VAL_Cost after epoch %i: %f' %( epoch + number, val_loss ) )



if __name__ == '__main__':
    args = parse_args()
    FLAGS = read_config.read_config_file( args.conf )
    main( FLAGS )