import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

from skimage import transform, exposure

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

def tile_alt_imshow(img_arrays, heat_maps=None, labels=None, titles=None, label_choice=1,
                    width=40, height=40, save_it=None, h_slot=None, w_slot=None,
                    cmap='jet', alpha=0.3, vmin=None, vmax=None, colorbar=False,
                    prob_array=None, force_single_channel=False, pat_boundaries=None, show_rl=True):

    plot_heat_map = None

    #scaled_img_arrays = rescale_img(img_arrays)
    scaled_img_arrays = img_arrays

    if len(img_arrays.shape) == 4:
        img_n, img_h, img_w, _ = img_arrays.shape
    else:
        img_n, img_h, img_w = img_arrays.shape

    if h_slot is None:
        h_slot = int(math.ceil(np.sqrt(img_n)))
    if w_slot is None:
        w_slot = int(math.ceil(np.sqrt(img_n)))
    if (h_slot == 1) & (w_slot == 1):
        w_slot=2 # if only 1 img

    fig, axes = plt.subplots(h_slot, w_slot, figsize=(width, height))
    fig.subplots_adjust(hspace=0.1, wspace=0)

    for ax, i in zip(axes.flatten(), range(img_n)):
        #img = rescale_img(scaled_img_arrays[i])
        img = scaled_img_arrays[i]
        #p2, p98 = np.percentile(img, (2, 98))
        #img = exposure.rescale_intensity(img, in_range=(p2, p98))
        img = exposure.equalize_hist(img)

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

        if heat_maps is not None:
            resized_map = transform.resize(heat_maps[i], (224, 224), mode='reflect', anti_aliasing=True)
            # resized_map = rescale_img(resized_map)
            plot_heat_map = ax.imshow(resized_map, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)

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
        plt.savefig(str(save_it), dpi=300, format='png')
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