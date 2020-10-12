import numpy as np
import skimage
import math

from .model_tools import load_densenet

def np_preprocess(uint16=False):
    if not uint16:
        pixel_max = 255.
        imgs_as = skimage.img_as_ubyte
    else:
        pixel_max = 65535.
        imgs_as = skimage.img_as_uint

    def inner_fxn(img, label):
        # for densenet, mode='torch'
        img = imgs_as(img)
        img = np.float32(img)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        c0 = (img[..., 0] / pixel_max - mean[0]) / std[0]
        c1 = (img[..., 1] / pixel_max - mean[0]) / std[1]
        c2 = (img[..., 2] / pixel_max - mean[0]) / std[2]
        new_img = np.stack([c0, c1, c2], axis=-1)

        return new_img, label

    return inner_fxn

# pos_label = {0: ap, 1: ll, 2: lo, 3: ls, 4: rl, 5: ro, 6: rs}
def label_with_cat_model(npz_path, model_weights, uint16=True, n_class=7, file_name=None, steps=1000):

    with np.load(npz_path) as f:
        image_array = f['image_array']
        index_array = f['index_array']

    model = load_densenet(input_size=image_array.shape[1], n_class=n_class)
    model.load_weights(model_weights).expect_partial()
    # suppress warning about not loading all params such as optimizer
    # no warning when run piecemeal in jupyter

    conc_list = []
    init_val = 0
    steps = steps
    iter_n = math.ceil(image_array.shape[0]/steps)

    for i in range(iter_n):
        c_imgs = image_array[init_val:init_val+steps]
        c_idx = index_array[init_val:init_val+steps]
        init_val+=steps
        c_imgs = c_imgs[..., None] + np.zeros((1, 1, 1, 3))
        p_imgs, p_idx  = np_preprocess(uint16)(c_imgs, c_idx)
        outs = model.predict(p_imgs)
        if n_class > 1:
            outs = list(np.argmax(outs, axis=1))
        conc_list = conc_list + outs
    labels = np.array(conc_list)

    if file_name is not None:
        np.savez(file_name, image_array=image_array, index_array=index_array, label_array=labels)

    return image_array, index_array, labels