from PIL import Image
import numpy as np
import tensorflow as tf
import os
import easygui

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


tfrecords_filename = input('Please input requested filename: ') + '.tfrecords'
writer = tf.python_io.TFRecordWriter(tfrecords_filename)

path_to_images = easygui.diropenbox(msg='Please select image directory', title=None, default=None)

for f in os.listdir(path_to_images):
    img = np.array(Image.open(os.path.join(path_to_images,f)))

    img_raw = img.tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'image_raw': _bytes_feature(img_raw)}))

    writer.write(example.SerializeToString())

writer.close()