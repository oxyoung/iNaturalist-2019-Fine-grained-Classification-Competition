import sys
sys.path.insert(0, '/home/oxyoung/ai_mecha/')
from GeneralTools.misc_fun import FLAGS
FLAGS.DEFAULT_IN = '/data/cephfs/punim0811/Datasets/iNaturalist/tfrecords_299/'
# FLAGS.DEFAULT_DOWNLOAD = '/data/cephfs/punim0811/'
FLAGS.DEFAULT_OUT = '/home/oxyoung/ai_mecha/Results/iNature'

FLAGS.IMAGE_FORMAT = 'channels_last'
FLAGS.IMAGE_FORMAT_ALIAS = 'NHWC'
from GeneralTools.inaturalist_func import ReadTFRecords
import os
import tensorflow as tf
from PIL import Image
import numpy as np

batch_size = 2
target_size = 299
key = 'val'
data_size = {'train': 265213, 'val': 3030, 'test': 35350}
data_label = {'train': 1, 'val': 1, 'test': 0}
num_images = data_size[key]
skip_count = num_images % batch_size
num_labels = data_label[key]

filenames = os.listdir(FLAGS.DEFAULT_IN)
filenames = [filename.replace('.tfrecords', '') for filename in filenames if key in filename]
print(filenames)

dataset = ReadTFRecords(
    filenames, num_labels=num_labels, batch_size=1,
    skip_count=skip_count, num_threads=8, decode_jpeg=True)
dataset.shape2image(3, target_size, target_size)
data_batch = dataset.next_batch()

with tf.Session() as sess:
    if key == 'test':
        x = sess.run(data_batch['x'])
    else:
        x, y = sess.run([data_batch['x'], data_batch['y']])

# visualize one sample from the batch
x_im = (x[0] + 1.0) * 127.5
im = Image.fromarray(x_im.astype(np.uint8), 'RGB')
im.save('test_image.jpg', 'JPEG')