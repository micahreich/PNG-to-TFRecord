from PIL import Image
import numpy as np
import tensorflow as tf
import os


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


tfrecords_filename = 'image001.tfrecords'

writer = tf.python_io.TFRecordWriter(tfrecords_filename)

#Path to images from dir
path_to_images = 'images_animation'

#List of images - method of accessing images
filenum = len([name for name in os.listdir(path_to_images) if os.path.isfile(os.path.join(path_to_images, name))])

for p in range(1, filenum):
    fname = "images_animation/image%03d.png" % p
    img = np.array(Image.open(fname))

    img_raw = img.tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'image_raw': _bytes_feature(img_raw)}))

    writer.write(example.SerializeToString())

writer.close()