"""
This code is modified based on richard 'Week 11 - A Working Example for Project 2 on Local Machine
--misc_fun.py'.

This file contains FLAGS definitions.
"""
import tensorflow as tf
flags = tf.flags
FLAGS = tf.flags.FLAGS

# This is a bug of Jupyter notebook
flags.DEFINE_string('f', '', "Empty flag to suppress UnrecognizedFlagError: Unknown command line flag 'f'.")

# local machine configuration
flags.DEFINE_integer('NUM_GPUS', 1, 'Number of GPUs in the local system.')
flags.DEFINE_float('EPSI', 1e-10, 'The smallest positive number to consider.')
flags.DEFINE_bool('ALLOW_GROWTH', False, 'Allow gpu memory to grow, for debugging purpose.')
flags.DEFINE_bool('MIXED_PRECISION', True, 'Use TensorFlow automatic mixed precision.')
flags.DEFINE_bool('XLA_JIT', False, 'Use TensorFlow Accelerated Linear Algebra in Just-in-times compilation.')

# library info
flags.DEFINE_string('TENSORFLOW_VERSION', '1.13.0', 'Version of TensorFlow for the current project.')
flags.DEFINE_string('CUDA_VERSION', '10.0', 'Version of CUDA for the current project.')
flags.DEFINE_string('CUDNN_VERSION', '7.3', 'Version of CuDNN for the current project.')

# working directory info
flags.DEFINE_string('SYSPATH', 'C:/Users/oxyoung/Desktop/AI Assignment2', 'Default working folder.')
flags.DEFINE_string('DEFAULT_IN', 'C:/Users/oxyoung/Desktop/AI Assignment2/download/', 'Default input folder.')
flags.DEFINE_string('DEFAULT_OUT', 'C:/Users/oxyoung/Desktop/AI Assignment2/Results/iNaturalist_ckpt/', 'Default output folder.')
flags.DEFINE_string(
    'DEFAULT_DOWNLOAD', 'C:/Users/oxyoung/Desktop/AI Assignment2/',
    'Default folder for downloading large datasets.')
flags.DEFINE_string(
    'INCEPTION_V3',
    'C:/Users/oxyoung/Desktop/AI Assignment2/code/inception_v1/inceptionv1_for_inception_score.pb',
    'Folder that stores InceptionV3 model.')

# data format
flags.DEFINE_string('IMAGE_FORMAT', 'channels_first', 'The format of images by default.')
flags.DEFINE_string('IMAGE_FORMAT_ALIAS', 'NCHW', 'The format of images by default.')

# model hyper-parameters
flags.DEFINE_string(
    'WEIGHT_INITIALIZER', 'default',
    'The default weight initialization scheme. Could also be sn_paper, pg_paper')
flags.DEFINE_string(
    'SPECTRAL_NORM_MODE', 'default',
    'The default power iteration method. Default is to use PICO. Could also be sn_paper.')
flags.DEFINE_bool('VERBOSE', True, 'Define whether to print more info during training and test.')
