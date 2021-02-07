import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import skimage
import tensorflow_addons as tfa
import tensorflow as tf

from skimage import transform, exposure

def derotate(image_array, label_array):
    img_holder = []
    for img, float_label in zip(image_array, label_array):
        img_holder.append(tfa.image.rotate(skimage.img_as_float(img), -float_label * tf.cast(math.pi / 180, tf.float32)).numpy())
    img_holder = np.array(img_holder)
    return img_holder

def deg_derotate(image_array, label_array, flip=None):
    if flip is None:
        flip = [False]*image_array.shape[0]
    img_holder = []
    for img, deg_label, flip_label in zip(image_array, label_array, flip):
        _img = transform.rotate(img, deg_label, resize=False)
        if flip_label:
            _img = np.flip(_img, axis=1)
        img_holder.append(_img)
    img_holder = np.array(img_holder)
    return img_holder

def resize_label_array(label_array, size_tuple):
    old_h = label_array.shape[0]
    old_w = label_array.shape[1]
    old_c = label_array.shape[2]

    new_h = size_tuple[0]
    new_w = size_tuple[1]
    resized_array = np.zeros((new_h, new_w, old_c))

    # With math.ceil, the boxes are going to be shifted down and to the right.  Consider a shape that is firmly divisible between the new and old shapes (300 by 20 is split evenly to 15; 300 by 12 is split to 25).  Or consider alternating between ceil and floor.
    h_ratio = math.ceil(new_h / old_h)
    w_ratio = math.ceil(new_w / old_w)

    for channel in range(old_c):
        for i in range(old_h):
            curr_row = i * h_ratio

            for j in range(old_w):
                curr_column = j * w_ratio
                resized_array[curr_row: (curr_row + h_ratio), curr_column : (curr_column + w_ratio), channel] = label_array[i, j, channel]

    return resized_array

def multiple_auc(one_dict, two_dict, third_dict=None, save_it=None, dpi=300):
    if third_dict is not None:
        n=3
    else:
        n=2
    f, axes = plt.subplots(1, n, figsize=(int(n*15), 15))
    plot_aucs(one_dict, ax=axes[0], title='View ROC curve')
    plot_aucs(two_dict, ax=axes[1], int_labels=['0', '1'], str_labels=['No hardware', 'Positive hardware'], title='Hardware ROC curve')
    if third_dict is not None:
        plot_aucs(third_dict, ax=axes[2], int_labels=['0', '1', '2'], str_labels=['Neutral', 'Flexion', 'Extension'], title='Dynamic position ROC curve')

    if save_it is not None:
        plt.savefig(str(save_it), dpi=dpi, format='tiff', bbox_inches='tight', pad_inches=0.1)

def plot_aucs(out_vals_dict, int_labels=None, title=None, str_labels=None, ax=None, save_it=None): #get_out_values from calc_metrics
    if int_labels is None:
        # int_labels = [0, 1, 2, 3, 4, 5, 6, 7]
        int_labels = [0, 1, 2, 3, 4, 5, 6]
    if str_labels is None:
        # str_labels = ['ap', 'll', 'lo', 'ls', 'rl', 'ro', 'rs', 'error']
        str_labels = ['Anterior-posterior', 'Left lateral', 'Left oblique', 'Left lumbosacral', 'Right lateral', 'Right oblique', 'Right lumbosacral']

    if ax is None:
        f, ax = plt.subplots()
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    if title is None:
        title = 'Receiver operator characteristic curve'
    ax.set_title(title)

    axin = ax.inset_axes([0.4, 0.4, 0.45, 0.45])
    x1, x2, y1, y2 = -0.01, 0.1, 0.9, 1.01
    axin.set_xlim(x1, x2)
    axin.set_ylim(y1, y2)
    # axin.set_xticks([0, 0.1, 0.2])
    # axin.set_yticks([0.8, 0.9, 1.0])
    ax.indicate_inset_zoom(axin, label=None)

    for int_label, cname in zip(int_labels, str_labels):
        ax.plot(out_vals_dict[int_label]['fpr'], out_vals_dict[int_label]['tpr'], label=cname)
        axin.plot(out_vals_dict[int_label]['fpr'], out_vals_dict[int_label]['tpr'])

    ax.legend(loc=4)
    if save_it is not None:
        plt.savefig(str(save_it), dpi=100, format='tiff', bbox_inches='tight', pad_inches=0)


def tile_alt_imshow(img_arrays, heat_maps=None, labels=None, titles=None, label_choice=1,
                    width=40, height=40, save_it=None, h_slot=None, w_slot=None, hspace=0, wspace=0,
                    cmap='jet', alpha=0.3, vmin=None, vmax=None, colorbar=False, dpi=100, axis_title_font=30,
                    prob_array=None, force_single_channel=False, pat_boundaries=None, show_rl=True, axis_titles=None,
                    ):

    plot_heat_map = None

    if len(img_arrays.shape) == 4:
        img_n, img_h, img_w, _ = img_arrays.shape
    else:
        img_n, img_h, img_w = img_arrays.shape

    if img_arrays.dtype == np.uint16(1).dtype: # from uint16 to uint8 for plt
        img_arrays = skimage.img_as_ubyte(img_arrays)

    if h_slot is None:
        h_slot = int(math.ceil(np.sqrt(img_n)))
    if w_slot is None:
        w_slot = int(math.ceil(np.sqrt(img_n)))
    if (h_slot == 1) & (w_slot == 1):
        w_slot=2 # if only 1 img

    fig, axes = plt.subplots(h_slot, w_slot, figsize=(width, height))
    fig.subplots_adjust(hspace=hspace, wspace=wspace)
    # fig.text(.15, .85, 'hello', bbox={'facecolor': 'white', 'pad': 2}, fontsize=30,
    # verticalalignment='top', horizontalalignment='left')

    #scaled_img_arrays = rescale_img(img_arrays)
    scaled_img_arrays = img_arrays

    for ax, i in zip(axes.flatten(), range(img_n)):
        #img = rescale_img(scaled_img_arrays[i])
        img = scaled_img_arrays[i]
        #p2, p98 = np.percentile(img, (2, 98))
        #img = exposure.rescale_intensity(img, in_range=(p2, p98))
        img = exposure.equalize_hist(img)
        img_dims = img.shape[:2]

        if labels is not None:
            c_labels = resize_label_array(labels[i], (img_h, img_w))
            img *= np.expand_dims(c_labels[..., label_choice], axis=2)

        if force_single_channel:
            cxr = ax.imshow(img[...,0], cmap='gray')
        else:
            cxr = ax.imshow(img, 'gray')

        if prob_array is not None:
            c_prob = prob_array[i]
            c_text = 'cls_0: {0:.2f}\ncls_1: {1:.2f}\ncls_2: {2:.2f}\ncls_3: {3:.2f}'.format(c_prob[0], c_prob[1], c_prob[2], c_prob[3])
            ax.text(10, 30, c_text, bbox={'facecolor': 'white', 'pad': 2})

        if axis_titles is not None:
            c_title = axis_titles[i]
            ax.text(0.01, 0.01, c_title, bbox={'facecolor': 'white', 'pad': 2}, fontsize=axis_title_font,
                    verticalalignment='bottom', horizontalalignment='left', transform=ax.transAxes)

        if heat_maps is not None:
            c_heatmap = heat_maps[i]
            if c_heatmap.shape[:2] != img_dims: # make sure (h,w) is same betwee img and heatmap
                c_heatmap = transform.resize(c_heatmap, img_dims, mode='reflect', anti_aliasing=True)
            plot_heat_map = ax.imshow(c_heatmap, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)

        if show_rl:
            ax.text(int(0.05*img_w), img_h//2, 'RIGHT', fontsize=10, color='red')
            ax.text(int(0.7*img_w), img_h//2, 'LEFT', fontsize=10, color='red')

        if titles is not None:
            c_title = titles[i]
            if pat_boundaries is not None:
                if pat_boundaries[i]:
                    c_title = '***** NEW PATIENT *****:  ' + str(c_title)

            ax.set_title(c_title, color='red')

        ax.axis('off')

        if img_n == 1: # if only 1 image
            break

    if colorbar:
        fig.colorbar(plot_heat_map)

    if save_it is not None:
        plt.savefig(str(save_it), dpi=dpi, format='tiff', bbox_inches='tight', pad_inches=0)
    else:
        plt.show()

def iterate_tile(image_arrays, label_arrays=None, label_choice=None, titles=None, start_i=0, custom_n=None, save=False, pat_boundaries=None):
    if label_choice is not None:
        c_array = image_arrays[label_arrays == label_choice]
    else:
        c_array = image_arrays

    if custom_n is None:
        n = np.ceil(c_array.shape[0] / 100).astype(np.int64)
    else:
        n = custom_n

    if titles is None:
        titles = np.arange(start_i + 100 * n)
    elif label_choice is not None:
        titles = titles[label_arrays == label_choice]
    else:
        titles = titles

    if pat_boundaries is None:
        pat_boundaries = [False] * (start_i + 100 * n)

    save_it = None
    for i in range(n):
        if save:
            save_it = i
        sliced_array = c_array[(start_i + i*100): (start_i + (i+1)*100)]
        sliced_titles = titles[(start_i + i*100): (start_i + (i+1)*100)]
        sliced_boundaries = pat_boundaries[(start_i + i*100): (start_i + (i+1)*100)]
        tile_alt_imshow(sliced_array, titles=sliced_titles, save_it=save_it, pat_boundaries=sliced_boundaries)
        print('\n')

def rescale_img(img, new_min=0, new_max=1):
    # not perfect since old min and old max based on curr img, and not on whole dataset
    # works WELL when used on array of images
    return (img - np.min(img)) * (new_max - new_min) / (np.max(img) - np.min(img)) + new_min


def save_img_as_jpg(_img, c_name):
    _img = exposure.equalize_hist(_img)
    if len(_img.shape) == 2:
        h, w = _img.shape
    else:
        h, w, _ = _img.shape
    fig_h = h / 1000
    fig_w = w / 1000
    fig = plt.figure(frameon=False)
    fig.set_size_inches(fig_w, fig_h)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(_img, aspect='auto', cmap='gray')
    fig.savefig(c_name, dpi=1000, format='jpeg')
    plt.close(fig)

# pixel size is same as input.  Works for mpl 3.0.1, 3.0.3 and 3.2.1
def save_inp_as_output(_img, c_name, dpi=100):
    h, w, _ = _img.shape
    fig, axes = plt.subplots(figsize=(h/dpi, w/dpi))
    _img = skimage.img_as_ubyte(_img)
    _img = exposure.equalize_hist(_img)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0) # axes fills figure
    axes.imshow(_img)
    axes.axis('off')
    plt.savefig(c_name, dpi=dpi, format='jpeg')
    # no bbox_inches or pad_inches - creates whitespace with mpl 301 and 303.
    # with mpl 321, bbox_inches creates padding and pad_inches saves blank image.