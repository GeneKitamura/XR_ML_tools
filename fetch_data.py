

import pandas as pd
import numpy as np
import tensorflow as tf
import skimage

from glob import glob
from .TFRecording import read_TFR_from_array
from .model_tools import trunc_name


AUTOTUNE = tf.data.experimental.AUTOTUNE

# will only use values from df_excel with rotation angles
# manually changed relook cases for './pos_hard_processed/rot_labeled_0to3945.xlsx'
def data_get(df_excel, label_col, train_bool_col, index_name='index',
             train_npz='./pos_hard_npz/238_0to3495.npz', val_npz='./pos_hard_npz/224_0to3495.npz',
             img_as_uint16=True, flip_RtoL_col=None, npz_df=None,
             df_start=0, df_end=None, expand_channels=True, test_set_col=None):

    if flip_RtoL_col is not None:
        label_df = pd.read_excel(npz_df) # just for position of right vs. left, accounts for whole df/npz
        right_bool = label_df[flip_RtoL_col].isin([4, 5, 6]).to_numpy()

    df = pd.read_excel(df_excel, index_col=index_name) # df only containing ll/rl with rot values
    df = df[df.index>df_start]
    if df_end is not None:
        df = df[df.index<df_end]

    df[train_bool_col] = df[train_bool_col].map(bool)

    n_labels = df[label_col].nunique()

    train_idx = df[df[train_bool_col]].index.tolist()
    train_idx = np.array(train_idx)
    val_idx = df[~df[train_bool_col]].index.tolist()
    val_idx = np.array(val_idx)

    train_labels = df.loc[train_idx, label_col].to_numpy()
    val_labels = df.loc[val_idx, label_col].to_numpy()

    with np.load(train_npz) as f: #img_as_float
        train_images = f['image_array']
        train_indices = f['index_array']
    if flip_RtoL_col is not None:
        flip_train_imgs = np.flip(train_images, axis=2)
        train_images = np.where(right_bool[..., None, None], flip_train_imgs, train_images)

    if img_as_uint16: # Memory intensive
        train_images = skimage.img_as_uint(train_images)
    else:
        train_images = skimage.img_as_ubyte(train_images)
    dtype = train_images.dtype

    with np.load(val_npz) as f: #img_as_float
        val_images = f['image_array']
        val_indices = f['index_array']
    if flip_RtoL_col is not None:
        flip_train_imgs = np.flip(val_images, axis=2)
        val_images = np.where(right_bool[..., None, None], flip_train_imgs, val_images)

    if img_as_uint16:
        val_images = skimage.img_as_uint(val_images)
    else:
        val_images = skimage.img_as_ubyte(val_images)

    if val_images.shape[0] != df.shape[0]: #some df from whole npz
        train_arange_df = pd.DataFrame({'index': np.arange(val_images.shape[0]), 'bool': False}).set_index('index')
        val_arange_df = train_arange_df.copy()

        train_arange_df.loc[train_idx] = True
        np_train_idx = train_arange_df['bool']
        val_arange_df.loc[val_idx] = True
        np_val_idx = val_arange_df['bool']
    elif df[-1:].index.values[0] == val_images.shape[0]: # every img/label on monotonically increasing df
        np_train_idx = train_idx
        np_val_idx = val_idx
    else: # every img/label on discontinuous df
        arange_idx = np.arange(df.shape[0])
        train_bool = df[train_bool_col]
        np_train_idx = arange_idx[train_bool]
        np_val_idx = arange_idx[~train_bool]

    train_images = train_images[np_train_idx]
    val_images = val_images[np_val_idx]

    # either way to create third channel; None is alias for np.newaxis()
    if expand_channels:
        # image_array = np.repeat(image_array[..., None], 3, axis=-1)
        train_images = train_images[..., None] + np.zeros((1, 1, 1, 3))
        train_images = train_images.astype(dtype)

        val_images = val_images[..., None] + np.zeros((1, 1, 1, 3))
        val_images = val_images.astype(dtype)

    train_indices = train_indices[np_train_idx]
    val_indices = val_indices[np_val_idx]

    test_images = None
    test_labels = None
    test_idx = None
    test_indices = None

    if test_set_col is not None: # train:0, val:1, test:2
        non_train_df = df.loc[val_idx] # val_df
        true_val_bool = non_train_df[test_set_col] == 1
        true_val_bool = true_val_bool.to_numpy() # true val bool.  test bool is ~true_val_bool

        # slices are by reference.  Even when variable is reassigned.
        # eg. a = np.arange(10)
        # b = a[5:]
        # a = a[7:]
        # a[2] = 11
        # b is [5,6,7,8,11]

        test_images = val_images[~true_val_bool]
        test_labels = val_labels[~true_val_bool]
        test_idx = val_idx[~true_val_bool]
        test_indices = val_indices[~true_val_bool]

        val_images = val_images[true_val_bool]
        val_labels = val_labels[true_val_bool]
        val_idx = val_idx[true_val_bool]
        val_indices = val_indices[true_val_bool]

    out_tup = (train_images, train_labels, train_idx, train_indices,
               val_images, val_labels, val_idx, val_indices,
               test_images, test_labels, test_idx, test_indices,
               n_labels)

    return out_tup


def prepare_dataset(*args, **kwargs):
    (train_images, train_labels, train_idx, train_indices,
     val_images, val_labels, val_idx, val_indices,
     test_images, test_labels, test_idx, test_indices,
     n_labels) = data_get(*args, **kwargs) # as ubyte

    test_ds = None
    n_test = None

    if test_images is not None:
        test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels, test_idx))
        n_test = test_images.shape[0]

    n_train = train_images.shape[0]
    n_val = val_images.shape[0]
    print('n_labels is {}'.format(n_labels))

    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels, train_idx))
    val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels, val_idx))

    return (train_ds, n_train, val_ds, n_val, test_ds, n_test, n_labels)

def TFR_dataset(train_dir, val_dir, n_train, n_val, n_labels, test_dir=None, n_test=None, array_dtype=np.uint16, train_size=660, val_size=600):

    train_files = glob(train_dir)
    len_train = len(train_files)
    val_files = glob(val_dir)
    len_val = len(val_files)
    test_files = None
    len_test = None

    train_ds = tf.data.Dataset.from_tensor_slices(train_files)
    val_ds = tf.data.Dataset.from_tensor_slices(val_files)
    test_ds = None

    train_decode_tfr = read_TFR_from_array(train_size, array_dtype=array_dtype)
    val_decode_tfr = read_TFR_from_array(val_size, array_dtype=array_dtype)

    # need n for interleave cycle_length (number of TFR files, NOT total n of images)
    train_ds = train_ds.interleave(lambda x: train_decode_tfr(x), cycle_length=len_train, block_length=1, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.interleave(lambda x: val_decode_tfr(x), cycle_length=len_val, block_length=1, num_parallel_calls=AUTOTUNE)
    #return decoded_img, label

    if test_dir is not None:
        test_files = glob(test_dir)
        len_test = len(test_files)
        test_ds = tf.data.Dataset.from_tensor_slices(test_files)
        test_ds = test_ds.interleave(lambda x: val_decode_tfr(x), cycle_length=len_test, block_length=1, num_parallel_calls=AUTOTUNE)

    return (train_ds, n_train, val_ds, n_val, test_ds, n_test, n_labels)

