import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn import preprocessing, metrics
from collections import namedtuple

def get_out_values(predictions, ind_labels, ids, class_int=None, class_str=None, sparse_labels=True):
    if class_int is None:
        class_int = [0, 4]
    if class_str is None:
        class_str = ['Normal', 'Prox femur']

    if not sparse_labels:
        ind_labels = np.argmax(ind_labels, axis=1)

    ind_predictions = np.argmax(predictions, axis=1)

    out_lib = {}

    # Accuracy as a whole (class 0,1,2,...) vs as binary in loop (class 0 vs not-0)
    n = predictions.shape[0]
    accuracy = metrics.accuracy_score(y_true=ind_labels, y_pred=ind_predictions)
    acc_CI = 1.96 * math.sqrt((accuracy * (1 - accuracy)) / n)
    print('total accuracy {:.3f} +- {:.3f}\n'.format(accuracy, acc_CI))

    # This method looks at class of interest vs. all others (4 vs [0,1,2,3,5])
    for i, j in zip(class_int, class_str):
        class_indicator = np.where(ind_labels == i, 1, 0)
        prediction_class_indicator = np.where(ind_predictions == i, 1, 0)
        fpr, tpr, thresholds = metrics.roc_curve(y_true=class_indicator, y_score=predictions[..., i])
        n_pos = class_indicator[class_indicator == 1].shape[0]
        n_others = class_indicator[class_indicator == 0].shape[0]

        auc_val = metrics.auc(fpr, tpr)
        q0 = auc_val * (1 - auc_val)
        q1 = auc_val / (2 - auc_val) - (auc_val ** 2)
        q2 = ((2 * auc_val ** 2) / (1 + auc_val)) - (auc_val ** 2)
        se = math.sqrt((q0 + (n_pos - 1) * q1 + (n_others - 1) * q2) / (n_pos * n_others))
        auc_95_CI = 1.96 * se
        print('{} auc {:.3f} +- {:.3f}'.format(j, auc_val, auc_95_CI))

        print('argmax_values:')
        _ = sample_metrics(prediction_class_indicator, class_indicator, print_out=True)

        thresh_val = show_metrics(predictions[..., i], class_indicator, thresholds)
        print('\n')

        post_threshold = np.array(predictions[..., i] > thresh_val, np.int)

        class_filter = (ind_labels == i)

        i_name = str(i)
        out_lib[i_name + '_out'] = post_threshold[class_filter]
        out_lib[i_name + '_ids'] = ids[class_filter]
        out_lib[i_name + '_auc'] = auc_val
        out_lib[i_name + '_auc_SE'] = se

    return out_lib

def sample_metrics(predictions_argmax, labels_argmax, print_out=False):
    n = predictions_argmax.shape[0]
    eps = 1e-5
    tn, fp, fn, tp = metrics.confusion_matrix(labels_argmax, predictions_argmax).ravel()
    sens = tp / (tp + fn + eps)
    spec = tn / (tn + fp + eps)
    ppv = tp / (tp + fp + eps)
    npv = tn / (tn + fn + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)

    sens_ci = 1.96 * math.sqrt((sens * (1 - sens))/ n)
    spec_ci = 1.96 * math.sqrt((spec * (1 - spec))/ n)
    ppv_ci = 1.96 * math.sqrt((ppv * (1 - ppv))/ n)
    npv_ci = 1.96 * math.sqrt((npv * (1 - npv))/ n)
    acc_ci = 1.96 * math.sqrt((acc * (1 - acc))/ n)

    if print_out:
        print('Accuracy {:.3f} +- {:.3f}'.format(acc, acc_ci))
        print('Sensitivity {:.3f} +- {:.3f}.  Specificity of {:.3f} +- {:.3f}.'.format(sens, sens_ci, spec, spec_ci))
        print('PPV {:.3f} +- {:.3f} and NPV {:.3f} +- {:.3f}'.format(ppv, ppv_ci, npv, npv_ci))

    return (sens, spec, ppv, npv, acc, sens_ci, spec_ci, ppv_ci, npv_ci, acc_ci)


def show_metrics(predict_1, labels, thresh_1, single_man_thresh_val=None):

    labels_argmax = labels
    eps = 1e-5

    thresh_list = []
    sens_list = []
    spec_list = []
    ppv_list = []
    npv_list = []
    acc_list = []

    best_comb = 0
    best_thresh_idx = 0
    best_thres_val = 0
    best_sens = 0
    best_spec = 0
    best_ppv = 0
    best_npv = 0
    n = predict_1.shape[0]

    l_range = 0
    h_range = thresh_1.shape[0]
    for i in range(l_range, h_range):
        a_thresholded = np.array(predict_1 > thresh_1[i], np.int)
        (sens, spec, ppv, npv, acc, sens_ci, spec_ci, ppv_ci, npv_ci, acc_ci) = sample_metrics(labels_argmax, a_thresholded)

        comb_val = sens + spec
        if comb_val > best_comb:
            best_comb = comb_val
            best_thresh_idx = i
            best_thres_val = thresh_1[i]
            best_sens = sens
            best_spec = spec
            best_ppv = ppv
            best_npv = npv

        if single_man_thresh_val is not None:
            a_thresholded = np.array(predict_1 > single_man_thresh_val, np.int)
            (sens, spec, ppv, npv, acc, sens_ci, spec_ci, ppv_ci, npv_ci, acc_ci) = sample_metrics(labels_argmax, a_thresholded)
            best_sens = sens
            best_spec = spec
            best_ppv = ppv
            best_npv = npv
            best_thres_val = single_man_thresh_val
            break

        thresh_list.append(i)
        sens_list.append(sens)
        spec_list.append(spec)
        ppv_list.append(ppv)
        npv_list.append(npv)
        acc_list.append(acc)

        #print(i, sens, spec, ppv, npv, acc)

    best_sens_ci = 1.96 * math.sqrt((best_sens * (1 - best_sens))/ n)
    best_spec_ci = 1.96 * math.sqrt((best_spec * (1 - best_spec))/ n)
    best_ppv_ci = 1.96 * math.sqrt((best_ppv * (1 - best_ppv))/ n)
    best_npv_ci = 1.96 * math.sqrt((best_npv * (1 - best_npv))/ n)

    print('Best threshold val:', best_thres_val)
    print('Best threshold idx: {:d}. Sensitivity {:.3f} +- {:.3f}.  Specificity of {:.3f} +- {:.3f}.'.format(best_thresh_idx, best_sens, best_sens_ci, best_spec, best_spec_ci))
    print('PPV {:.3f} +- {:.3f} and NPV {:.3f} +- {:.3f}'.format(best_ppv, best_ppv_ci, best_npv, best_npv_ci))

    thresh_list = np.array(thresh_list)
    sens_list = np.array(sens_list)
    spec_list = np.array(spec_list)
    ppv_list = np.array(ppv_list)
    npv_list = np.array(npv_list)
    acc_list = np.array(acc_list)

    return best_thres_val
