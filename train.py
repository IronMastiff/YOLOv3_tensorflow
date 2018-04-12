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
    datas, labels = net.create_placeholder( FLAGS.width, FLAGS.height, scale_width, scale_height )

    '''--------net--------'''
    pre_scale1, pre_scale2, pre_scale3 = net.feature_extractor( datas )
    scale1, scale2, scale3 = net.scales( pre_scale1, pre_scale2, pre_scale3 )



    '''--------get labels_filenames and datas_filenames--------'''
    datas_filenames = reader.images( FLAGS.batch_size, FLAGS.datas_path )
    labels_fienames = reader.labels( FLAGS.batch_size, FLAGS.labels_path )
    normalize_labels = extract_labels.labels_normaliszer( labels_fienames,
                                                          FLAGS.width,
                                                          FLAGS.height,
                                                          scale_width,
                                                          scale_height )

    '''---------partition the train data and val data--------'''
    train_filenames = datas_filenames[: int( len( datas_filenames ) * 0.9 )]
    train_labels = normalize_labels[: int( len( normalize_labels ) * 0.9 )]
    val_filenames = datas_filenames[len( datas_filenames ) - int( len( datas_filenames ) * 0.9 ) :]
    val_labels = normalize_labels[len( normalize_labels ) - int( len( normalize_labels ) * 0.9 ) :]

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        saver = tf.train.Saver()
        save_path = select_things.select_checkpoint( FLAGS.scale )
        last_checkpoint = tf.train.latest_checkpoint( save_path, 'checkpoint' )
        if last_checkpoint:
            saver.restore( sess, last_checkpoint )
        else:
            sess.run( init )


        for epoch in range( FLAGS.epoch ):
            epoch_loss = 0
            for i in range( len( train_filenames ) ):
                normalize_datas = []
                for data_filename in train_filenames[i]:
                    image = reader.get_image( data_filename, FLAGS.width, FLAGS.height )
                    image = np.array( image, np.float32 )

                    normalize_datas.append( image )

                normalize_datas = np.array( normalize_datas )

                scale1, scale2, scale3 = sess.run( [scale1, scale2, scale3], feed_dict = {datas: normalize_datas} )

                '''--------calculate loss--------'''
                if FLAGS.scale == 1:
                    loss = get_loss.calculate_loss( scale1, train_labels[i] )

                if FLAGS.scale == 2:
                    loss = get_loss.calculate_loss( scale2, train_labels[i] )

                if FLAGS.scale == 3:
                    loss = get_loss.calculate_loss( scale3, train_labels[i] )

                '''--------Optimizer--------'''
                optimizer = tf.train.AdamOptimizer( learning_rate=FLAGS.learning_rate ).minimize( loss )

                batch_loss = sess.run( [optimizer, loss] )

                epoch_loss =+ batch_loss

                tf.summary.scalar( 'epoch_loss', loss )

            if epoch % 10 == 0:
                print( 'Cost after epoch %i: %f' % ( epoch, epoch_loss ) )

            if epoch % 50 == 0:
                val_loss = 0
                for val_filename, val_label in val_filenames, val_labels:
                    normalize_datas = []
                    for data_filename in val_filename:
                        image = reader.get_image( data_filename, FLAGS.widht, FLAGS.height )
                        image = np.array( image, np.float32 )

                        normalize_datas.append( image )

                    normalize_datas = np.array( normalize_datas )

                    batch_loss = sess.run( [loss], feed_dict = {datas: normalize_datas, labels: val_label } )

                    val_loss =+ batch_loss

                    tf.summary.scalar('val_loss', val_loss)

                print( 'VAL_Cost after epoch %i: %f' %( epoch, val_loss ) )
                saver.save( sess, save_path, global_step = epoch )
        merged = tf.summary.merge_all()



if __name__ == '__main__':
    args = parse_args()
    FLAGS = read_config.read_config_file( args.conf )
    main( FLAGS )