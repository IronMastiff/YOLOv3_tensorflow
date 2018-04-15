import time
import tensorflow as tf
import os
import argparse
import cv2

import reader
from utils import net, read_config, select_things

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( '-c', '--conf', default = './config/eval_config.yml', help = 'the path to the eval_config file' )
    return parser.parse_args()

def main( FLAGS ):
    if not os.path.exists( FLAGS.save_dir ):
        os.makedirs( FLAGS.save_dir )


    input_image = reader.get_image( FLAGS.image_dir, FLAGS.image_width, FLAGS.image_height )




if __name__ == '__main__':
    args = parse_args()
    FLAGS = read_config.read_config_file( args.conf )
    main( FLAGS )