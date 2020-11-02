import numpy as np
import skimage
import math
import itertools

from scipy import stats

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
def label_with_cat_model(npz_path, model_weights, load_model, uint16=True, n_class=7, file_name=None, steps=1000):

    with np.load(npz_path) as f:
        image_array = f['image_array']
        index_array = f['index_array']

    model = load_model(input_size=image_array.shape[1], n_class=n_class)
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

def ensemble_scalar_outs(scalar_list=None, params=None, path_val=None, diff_pvals=True, print_out=False, sort_val=None, part_bool=None):
    if scalar_list is None:
        scalar_list = [20, 40, 60, 80, 100, 120, 140, 160, 180]

    if path_val is None:
        path_val = './rot/test_vals/{}'

    if params is None:
        params = {
            'labels': 'orig_labels',
            'predictions': 'scaled_pred_angle'
        }

    combs = list(itertools.combinations(scalar_list, 2))
    str_labels = params['labels']
    str_preds = params['predictions']

    single_dict = {}
    ensemble_dict = {}
    sorted_single = {}
    sorted_ensemble = {}
    just_pvals = {}

    with np.load(path_val.format(scalar_list[0]) + '.npz') as f:
        one_labels = f[str_labels]
    n = one_labels.shape[0]
    if part_bool is None:
        part_bool = [True] * n

    for i in scalar_list:
        one_path = path_val.format(i)
        with np.load(one_path + '.npz') as f:
            one_labels = f[str_labels][part_bool]
            one_preds = f[str_preds][part_bool]

        diff = np.abs(one_labels - one_preds)
        mean_diff = np.mean(diff)
        ci = 1.96 * np.std(diff) / np.sqrt(diff.shape[0])
        single_dict[i] = {'scalar_val': i, 'mean_diff': mean_diff, 'ci': ci}
        if print_out:
            print('{} MAE: {:.2f} +- {:.2f}'.format(i, mean_diff, ci))

    for i, j in combs:
        tmp_dict = {}
        one_path = path_val.format(i)
        with np.load(one_path + '.npz') as f:
            one_labels = f[str_labels][part_bool]
            one_preds = f[str_preds][part_bool]

        two_path = path_val.format(j)
        with np.load(two_path + '.npz') as f:
            two_labels = f[str_labels][part_bool]
            two_preds = f[str_preds][part_bool]

        one_diff = np.abs(one_labels - one_preds)
        two_diff = np.abs(two_labels - two_preds)
        if diff_pvals:
            p_val = stats.ttest_rel(one_diff, two_diff).pvalue
        else:
            p_val = stats.ttest_rel(one_preds, two_preds).pvalue
        tmp_dict['orig'] = {'p_val': p_val}
        just_pvals[(i, j)] = p_val
        if print_out:
            print('\ncurr vals {} and {} with p_val {:.4f}'.format(i, j, p_val))

        mean_vals = (one_preds + two_preds) / 2
        max_vals = np.maximum(one_preds, two_preds)
        min_vals = np.minimum(one_preds, two_preds)

        _stack_vals = np.stack([one_preds, two_preds], axis=1)
        _abs_max_ind = np.argmax(np.abs(_stack_vals), axis=1)
        abs_max_vals = np.where(_abs_max_ind == 0, one_preds, two_preds)
        abs_min_vals = np.where(_abs_max_ind == 0, two_preds, one_preds)  # flipped

        vals_mix_list = [mean_vals, max_vals, min_vals, abs_max_vals, abs_min_vals]
        mix_names = ['mean', 'max', 'min', 'abs_max', 'abs_min']

        assert np.mean(one_labels - two_labels) == 0, 'labels means different'

        for mix_val, mix_name in zip(vals_mix_list, mix_names):
            mix_diff = np.abs(one_labels - mix_val)
            if diff_pvals:
                one_p_val = stats.ttest_rel(one_diff, mix_diff).pvalue
                two_p_val = stats.ttest_rel(two_diff, mix_diff).pvalue
            else:
                one_p_val = stats.ttest_rel(one_preds, mix_val).pvalue
                two_p_val = stats.ttest_rel(two_preds, mix_val).pvalue
            mean_diff = np.mean(mix_diff)
            ci = 1.96 * np.std(mix_diff) / np.sqrt(mix_diff.shape[0])
            one_name = '{}_pval'.format(i)
            two_name = '{}_pval'.format(j)
            tmp_dict[mix_name] = {'mean_diff': mean_diff, 'ci': ci, one_name: one_p_val, two_name:two_p_val}
            if print_out:
                print('{} MAE: {:.2f} +- {:.2f}. {}: {:.4f}, {}: {:.4f}'.format(mix_name, mean_diff, ci, one_name, one_p_val, two_name, two_p_val))

        ensemble_dict[(i, j)] = tmp_dict

    if sort_val is not None:
        sorted_single = sorted(single_dict.items(), key=lambda x: x[1][sort_val], reverse=False)

        for val_tup, dict_items in ensemble_dict.items():
            for dict_key, dict_values in dict_items.items():
                if sort_val in dict_values:
                    sorted_ensemble[(*val_tup, dict_key)] = dict_values

        sorted_ensemble = sorted(sorted_ensemble.items(), key=lambda x: x[1][sort_val], reverse=False)

    return (single_dict, just_pvals, ensemble_dict, sorted_single, sorted_ensemble)

