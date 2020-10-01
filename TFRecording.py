import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import skimage

from .utils import array_min_max, array_print

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


def make_TFR_from_loaded_arrays(input_imgs, input_labels, input_names, save_path, strides=50, expand_channels=False):
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
        if expand_channels:
            dtype = c_imgs.dtype
            c_imgs = c_imgs[..., None] + np.zeros((1, 1, 1, 3))
            c_imgs = c_imgs.astype(dtype)
        c_labels = input_labels[start_n:end_n]
        c_names = input_names[start_n:end_n]

        end_name = '{}to{}'.format(start_n, end_n)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = os.path.join(save_path, end_name)

        with tf.io.TFRecordWriter(filename) as writer:
            for i in range(c_imgs.shape[0]):
                example = _serialize_img(c_imgs[i], c_labels[i], c_names[i])
                writer.write(example)

def read_TFR_from_array(out_shape, array_dtype=np.uint16): # need dtype for arrays

    def inner_fxn(tf_file):
        raw_dataset = tf.data.TFRecordDataset(tf_file)

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

        def _format_bytes(datapoint):
            img_as_string = datapoint['img']
            dtype = datapoint['dtype']
            label = datapoint['label']
            name = datapoint['name']
            decoded_img = tf.ensure_shape(tf.io.parse_tensor(img_as_string, array_dtype), (out_shape, out_shape, 3))
            # NOT Eager tensor, so cannot use parsed values to parse_tensor
            return decoded_img, label, name

        parsed_dataset = parsed_dataset.map(_format_bytes)
        return parsed_dataset

    return inner_fxn

def show_TFR_from_array(parsed_dataset):
    for datapoint in parsed_dataset.take(10): # now EAGER tensor
        decoded_img, label, name = datapoint
        decoded_img = decoded_img.numpy()
        label = datapoint['label'].numpy()
        name = datapoint['name'].numpy().decode('utf-8')
        array_min_max(decoded_img)
        print('name {} with label {}'.format(name, label))
        plt.imshow(skimage.img_as_ubyte(decoded_img), 'gray')

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

def show_images_from_files(parsed_dataset):
    for datapoint in parsed_dataset.take(1):
        img = tf.io.decode_image(datapoint['img'])
        label = datapoint['label'].numpy()
        name = datapoint['name'].numpy().decode('utf-8')
        f, ax = plt.subplots()
        ax.imshow(img)
        ax.set_title('{}_{}'.format(label, name))
        plt.show()