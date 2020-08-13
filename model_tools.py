import pandas as pd
import math
import tensorflow as tf
import time
import sys

sys.path.append('..')

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def split_train_val_df(labels_df, label_types=None, train_split=0.8):
    # labels_df with meaningful index to link back to whole df

    train_idx = {}
    if label_types is None:
        label_types = ['pos_label', 'hard_label']

    for label_type in label_types:
        train_idx[label_type] = []
        for label in labels_df[label_type].unique():
            c_df = labels_df[labels_df[label_type] == label]
            tn = int(c_df.shape[0] * train_split)
            tidx = c_df.sample(tn).index.tolist()
            train_idx[label_type] = train_idx[label_type] + tidx
        train_idx[label_type] = sorted(train_idx[label_type])

    return train_idx

def load_densenet(input_size=224, n_class=3):
    base_model = keras.applications.densenet.DenseNet121(include_top=False, input_shape=(input_size, input_size, 3), pooling='avg',  weights='imagenet')
    x = base_model.output  # (None, 1024)
    x = keras.layers.Dropout(0.5)(x)
    # x = keras.layers.Dense(n_class, activation='softmax', kernel_regularizer=keras.regularizers.l2())(x)
    x = keras.layers.Dense(n_class, kernel_regularizer=keras.regularizers.l2())(x)  # no softmax for stability
    model = Model(inputs=base_model.input, outputs=x)

    return model

def preprocess_densenet(img, label):
    # for densenet, mode='torch'
    img = tf.cast(img, tf.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    c0 = (img[..., 0] / 255. - mean[0]) / std[0]
    c1 = (img[..., 1] / 255. - mean[0]) / std[1]
    c2 = (img[..., 2] / 255. - mean[0]) / std[2]
    new_img = tf.stack([c0, c1, c2], axis=-1)

    return new_img, label

def position_augment(img, label, preprocess_fxn=preprocess_densenet):
    # data should be in uint8 or uint16 for brightness/contrast
    img = tf.image.random_brightness(img, 0.5)
    img = tf.image.random_contrast(img, 0.5, 1.5)
    img = tf.image.random_crop(img, (224, 224, 3))
    img = preprocess_fxn(img, label)
    return img, label

def augment_map(*datapoint):
    img, label = datapoint
    # img, label = position_augment(img, label, preprocess_fxn=lambda x, y: x,y) # to check images
    img, label = position_augment(img, label)

    return img, label

def train_numpy_keras(get_numpy_ds, batch_size=20, epochs=20, save_path=None, excel_path=None):
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_ds, n_train, val_ds, n_val = get_numpy_ds
    train_steps = math.ceil(n_train/batch_size)
    val_steps = math.ceil(n_val/batch_size)

    # to see items
    # iter_ds = iter(train_ds)
    # one_item = next(iter_ds)

    #shuffle MUST be after cache, otherwise data is always fed in the same way
    train_ds = train_ds.cache().map(augment_map).shuffle(buffer_size=1500).batch(batch_size).repeat().prefetch(AUTOTUNE)
    # no need to repeat val_ds; it will run from top every time.
    val_ds = val_ds.cache().batch(batch_size).prefetch(AUTOTUNE)

    model = load_densenet()

    optimizer = Adam(lr=1e-4)
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # can provide logits for SparseCategoricalAccuracy since argmax of logits and probs are the same
    model.compile(optimizer=optimizer, loss=loss, metrics=[keras.metrics.SparseCategoricalAccuracy()])

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-6, min_delta=1e-3),
        tf.keras.callbacks.EarlyStopping(patience=10, min_delta=1e-3),
        tf.keras.callbacks.ModelCheckpoint(filepath=save_path, save_best_only=True, save_weights_only=True)
    ]

    train_start = time.time()
    history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=callbacks, steps_per_epoch=train_steps, validation_steps=val_steps)

    print('total time {:.2f}'.format(time.time() - train_start), 3)

    c_df = pd.DataFrame()
    for key, val in history.history.items():
        c_df[key] = val
    c_df.to_excel(excel_path)