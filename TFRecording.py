import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import os

# tf.train.Feature is a ProtocolMessage
# tf.Example is a {'string': tf.train.Feature} mapping
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _array_feature(input_array):
    array_string = tf.io.serialize_tensor(tf.constant(input_array)).numpy() # don't need tf.constant.
    dtype = input_array.dtype.name.encode('utf-8')
    # to decode:
    # dtype = np.dtype(dtype)
    # parsed_string = tf.io.parse_tensor(array_string, dtype)
    return _bytes_feature(array_string), _bytes_feature(dtype)


def make_TFR_from_loaded_arrays(input_imgs, input_labels, input_names, save_path, strides=50):
    def _serialize_img(input_img, input_label, input_name):
        img_string, dtype_string = _array_feature(input_img)
        feature = {
            'img': img_string,
            'dtype': dtype_string,
            'label': _int64_feature(input_label),
            'name': _bytes_feature(input_name.encode('utf-8'))
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    cn = math.ceil(input_imgs.shape[0] / strides)

    for i in range(cn):

        start_n = i * strides
        end_n = (i + 1) * strides
        c_imgs = input_imgs[start_n:end_n]
        c_labels = input_labels[start_n:end_n]
        c_names = input_names[start_n:end_n]

        end_name = '{}to{}'.format(start_n, end_n)
        filename = os.path.join(save_path, end_name)

        with tf.io.TFRecordWriter(filename) as writer:
            for i in range(c_imgs.shape[0]):
                example = _serialize_img(c_imgs[i], c_labels[i], c_names[i])
                writer.write(example)

def read_TFR_from_array(): # need dtype for arrays
    raw_dataset = tf.data.TFRecordDataset('./tfr_sample')

    feature_description = {
        'img': tf.io.FixedLenFeature([], tf.string),
        'dtype': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'name': tf.io.FixedLenFeature([], tf.string)
    }

    def _parse_function(example_proto):
        # Parse the input `tf.Example` proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, feature_description)

    parsed_dataset = raw_dataset.map(_parse_function)

    for datapoint in parsed_dataset.take(1):
        img = datapoint['img']
        dtype = np.dtype(datapoint['dtype'].numpy())
        label = datapoint['label']
        parsed_string = tf.io.parse_tensor(img, dtype).numpy()

        plt.imshow(parsed_string)

def make_TFR_from_files(input_files, input_labels, file_name):
    def _serialize_img(input_img_bytes, input_label, input_name):
        feature = {
            'img': _bytes_feature(input_img_bytes),
            'label': _int64_feature(input_label),
            'name': _bytes_feature(input_name.encode('utf-8'))
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    with tf.io.TFRecordWriter(file_name) as writer:
        for i in range(len(input_files)):
            img = tf.io.read_file(input_files[i])
            example = _serialize_img(img, input_labels[i], input_files[i])
            writer.write(example)

def read_TFR_from_files(string_or_tfDataset): # if tf.io.read_file, don't need dtype
    raw_dataset = tf.data.TFRecordDataset(string_or_tfDataset)

    feature_description = {
        'img': tf.io.FixedLenFeature([], tf.string),
        'name': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }

    def _parse_function(example_proto):
        # Parse the input `tf.Example` proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, feature_description)

    parsed_dataset = raw_dataset.map(_parse_function)
    return parsed_dataset

def show_images_from_TFR(parsed_dataset):
    for datapoint in parsed_dataset.take(1):
        img = tf.io.decode_image(datapoint['img'])
        label = datapoint['label'].numpy()
        name = datapoint['name'].numpy().decode()
        f, ax = plt.subplots()
        ax.imshow(img)
        ax.set_title('{}_{}'.format(label, name))
        plt.show()