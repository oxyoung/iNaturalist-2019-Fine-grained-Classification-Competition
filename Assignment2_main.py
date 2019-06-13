"""
This code is modified based on richard 'Week 11 - A Working Example for Project 2 on Local Machine
--iNaturalist_main.py'.

In training mode, this code builds a CNN model, including InceptionV3, Xception and InceptionResNetV2 using Keras.
Then it can train and validates the model according to iNaturalist 2019 dataset and save the best model according
to validation accuracy.

In test mode, this code generates the submission CSV file for iNaturalist 2019 at FGVC6 competition.

If you want to run this code, please change all paths in this file according to your folder structure.

"""
# -----------------------------------------------Configuration----------------------------------------------------------
"""
Switch platform to run the current code: local, spartan.
"""
PLATFORM = 'local'
image_size = 512
import sys
import os
# File paths for local machine
if PLATFORM == 'local':
    sys.path.insert(0, 'C:/Users/oxyoung/Desktop/AI Assignment2')
    from GeneralTools.misc_fun import FLAGS
    FLAGS.DEFAULT_IN = 'C:/Users/oxyoung/Desktop/AI Assignment2/download/tfrecords_{}/'.format(image_size)
    FLAGS.DEFAULT_OUT = 'C:/Users/oxyoung/Desktop/AI Assignment2/Results/iNaturalist_ckpt/'
    FLAGS.DEFAULT_DOWNLOAD = 'C:/Users/oxyoung/Desktop/AI Assignment2/'
# File paths for spartan
elif PLATFORM == 'spartan':
    sys.path.insert(0, '/home/oxyoung/Assignment2')
    from GeneralTools.misc_fun import FLAGS
    FLAGS.DEFAULT_IN = '/data/cephfs/punim0811/Datasets/iNaturalist/tfrecords_{}/'.format(image_size)
    FLAGS.DEFAULT_OUT = '/home/oxyoung/Assignment2/Results/iNaturalist_ckpt/'
    FLAGS.DEFAULT_DOWNLOAD = '/data/cephfs/punim0811/Datasets/iNaturalist/'
else:
    raise OSError('The requested platform: {} has not been implemented.'.format(PLATFORM))

# Configure image format
FLAGS.IMAGE_FORMAT = 'channels_last'
FLAGS.IMAGE_FORMAT_ALIAS = 'NHWC'

# Import module and code
from GeneralTools.inaturalist_func import read_inaturalist
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras import applications
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.backend import set_session
import json
import numpy as np

# Check ALLOW_GROWTH and XLA_JIT
if FLAGS.ALLOW_GROWTH or FLAGS.XLA_JIT:
    config = tf.ConfigProto()
    # allow gpu memory to grow, for debugging purpose, safe to delete
    if FLAGS.ALLOW_GROWTH:
        config.gpu_options.allow_growth = True
        config.log_device_placement = False
    # use TensorFlow accelerated linear algebra
    if FLAGS.XLA_JIT:
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    sess = tf.Session(config=config)
    set_session(sess)

# --------------------------------------------End of Configuration------------------------------------------------------

# ---------------------------------------------------Model--------------------------------------------------------------
"""
Set default dataset image parameters.
"""
target_size = 299
num_classes = 1010
data_size = {'train': 265213, 'val': 3030, 'test': 35350}

"""
Define CNN model, including 'inceptionV3', 'Xception' and 'inception_resnet_v2'.
Also define training hyper-parameters, especially learning_rate, drop_out.
"""
model_name = 'Xception'
learning_rate = 1e-5
init_epoch = 0
target_epoch = 12
drop_out = 0.1
buffer_size = 768
"""
Switch between training mode and test mode by setting training_mode to True or False respectively.
do_save_and_load determine training from scratch or load pre-train model.
"""
training_mode = True
do_save_and_load = False

"""
Define the paths of three Pre-trained models on ImageNet for 'inceptionV3', 'Xception' and 'inception_resnet_v2'.
If train the model on spartan, the weights should be pre-downloaded for using, because spartan cannot download
the weights online, 
"""
inception_v3_path = 'C:/Users/oxyoung/Desktop/AI Assignment2/Pre-trained Model/' \
                    'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
Xception_path = 'C:/Users/oxyoung/Desktop/AI Assignment2/Pre-trained Model/' \
                'xception_weights_tf_dim_ordering_tf_kernels_notop.h5'
inception_resnet_v2_path = 'C:/Users/oxyoung/Desktop/AI Assignment2/Pre-trained Model/' \
                           'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'

"""
Prepare folder and ckpt files to load.
Here all model files follows the same format: ckpt_folder/ckpt_name_template.
In test mode, the target ckpt should be the only one in ckpt_folder to avoid mis-loading error.
"""
ckpt_folder = FLAGS.DEFAULT_OUT + 'trial_{}_{}_{}/'.format(image_size, target_size, drop_out)
if not os.path.isdir(ckpt_folder):
    os.mkdir(ckpt_folder)
ckpt_name_prefix = '{}_full_GAPDropD_{}_RMSProp_{}'.format(model_name, drop_out, learning_rate)
ckpt_list = [file for file in sorted(os.listdir(ckpt_folder)) if ckpt_name_prefix in file]
last_ckpt = ckpt_folder + (ckpt_list[-1] if ckpt_list else '')
ckpt_path_template = ckpt_folder + ckpt_name_prefix + '_{epoch:02d}_{val_acc:.3f}.h5'
if not training_mode:
    assert do_save_and_load, ValueError('In test mode, do_save_and_load must be true')
    assert os.path.isfile(last_ckpt), FileExistsError('The ckpt {} does not exist'.format(last_ckpt))

"""
Load the model from either a ckpt or a pre-trained model
"""
# load the model
if do_save_and_load and os.path.isfile(last_ckpt):
    mdl = load_model(
        last_ckpt,
        custom_objects={'softmax_cross_entropy': tf.losses.softmax_cross_entropy})
    mdl.compile(
        tf.keras.optimizers.RMSprop(lr=learning_rate),
        loss=tf.losses.softmax_cross_entropy, metrics=['accuracy'])
    print('Model loaded from {}'.format(last_ckpt))
else:
    print('Start training from scratch')
    if model_name == 'inceptionV3':
        base_model = applications.InceptionV3(
            weights=inception_v3_path,
            include_top=False,
            input_shape=(target_size, target_size, 3))
    elif model_name == 'Xception':
        base_model = applications.xception.Xception(include_top=False, weights=Xception_path, input_tensor=None,
                                       input_shape=(target_size, target_size, 3), pooling=None, classes=1000)
    elif model_name == 'inception_resnet_v2':
        base_model = applications.inception_resnet_v2.InceptionResNetV2(include_top=False,
            weights=inception_resnet_v2_path, input_tensor=None, input_shape=(target_size, target_size, 3),
            pooling=None, classes=1000)
    print('Model weights loaded from {}'.format(inception_v3_path))
    # Adding custom layers
    mdl = Sequential([
        base_model, GlobalAveragePooling2D('channels_last'), Dropout(drop_out),
        Dense(num_classes, activation='linear')])
    mdl.compile(
        tf.keras.optimizers.RMSprop(lr=learning_rate),
        loss=tf.losses.softmax_cross_entropy, metrics=['accuracy'])

mdl.summary()

"""
Do the training or prediction
"""
if training_mode:
    """
    Read the training and validation dataset. If train on spartan, recommended batch_size setting is:
    inceptionV3: batch_size 64
    Xception: batch_size 16
    inception_resnet_v2: batch_size 16
    """
    dataset_tr, steps_per_tr_epoch = read_inaturalist(
        'train', batch_size=16, image_size=image_size, target_size=target_size,
        do_augment=True, buffer_size=buffer_size)
    dataset_va, steps_per_va_epoch = read_inaturalist(
        'val', batch_size=16, image_size=image_size, target_size=target_size,
        do_augment=True, buffer_size=buffer_size)

    # Configure check point
    checkpoint = ModelCheckpoint(
        ckpt_path_template, monitor='val_acc', verbose=1,
        save_best_only=True, save_weights_only=False, mode='auto', period=1)

    # Start training
    start_time = time.time()
    history = mdl.fit(
        dataset_tr.dataset, epochs=target_epoch, initial_epoch=init_epoch, callbacks=[checkpoint],
        validation_data=dataset_va.dataset, steps_per_epoch=steps_per_tr_epoch, validation_steps=10, verbose=1)
    duration = time.time() - start_time
    print('\n The training process took {:.1f} seconds'.format(duration))

else:

    """
    Read the test dataset and start prediction
    """
    dataset_te, steps_per_te_epoch = read_inaturalist(
        'test', batch_size=101, image_size=image_size, target_size=target_size,
        do_augment=True, buffer_size=buffer_size)

    # Do the prediction
    start_time = time.time()
    y = mdl.predict(dataset_te.dataset, verbose=1, steps=steps_per_te_epoch)
    duration = time.time() - start_time
    print('\n The training process took {:.1f} seconds'.format(duration))

    # Get prediction labels
    labels = np.argmax(y, axis=1)
    # Get test image id from test2019.json file
    with open(os.path.join(FLAGS.DEFAULT_DOWNLOAD, 'test2019.json')) as data_file:
        image_annotations = json.load(data_file)
    images = image_annotations['images']
    images_id = np.array([image['id'] for image in images], dtype=int)
    save_data = np.concatenate(
        (np.expand_dims(images_id, axis=1), np.expand_dims(labels, axis=1)), axis=1)

    # Save to CSV file
    prediction_save_file = last_ckpt + '_predicted_labels.csv'
    print('Predictions saved to {}'.format(prediction_save_file))
    np.savetxt(prediction_save_file, save_data, fmt='%d', delimiter=',', header='id,predicted', comments='')

# -----------------------------------------------End of model-----------------------------------------------------------

