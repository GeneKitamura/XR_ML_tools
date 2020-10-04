import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn import preprocessing, metrics
from collections import namedtuple

#TODO: get metric outputs at argmax in end (PPV, sens, etc, but not AUC since that's one vs. all)
def get_out_values(predictions, ind_labels, ids, class_int=None, class_str=None, sparse_labels=True):
    if class_int is None:
        class_int = [0, 4]
    if class_str is None:
        class_str = ['Normal', 'Prox femur']

    if not sparse_labels:
        ind_labels = np.argmax(ind_labels, axis=1)

    ind_predictions = np.argmax(predictions, axis=1)

    n = predictions.shape[0]
    out_lib = {}

    # This method looks at class of interest vs. all others (4 vs [0,1,2,3,5])
    for i, j in zip(class_int, class_str):
        class_indicator = np.where(ind_labels == i, 1, 0)
        prediction_class_indicator = np.where(ind_predictions == i, 1, 0)
        fpr, tpr, thresholds = metrics.roc_curve(y_true=class_indicator, y_score=predictions[..., i])
        n_pos = class_indicator[class_indicator == 1].shape[0]
        n_others = class_indicator[class_indicator == 0].shape[0]

        accuracy = metrics.accuracy_score(y_true=class_indicator, y_pred=prediction_class_indicator)
        acc_CI = 1.96 * math.sqrt((accuracy * (1 - accuracy)) / n)
        print('{} accuracy {:.3f} +- {:.3f}'.format(j, accuracy, acc_CI))
        auc_val = metrics.auc(fpr, tpr)
        q0 = auc_val * (1 - auc_val)
        q1 = auc_val / (2 - auc_val) - (auc_val ** 2)
        q2 = ((2 * auc_val ** 2) / (1 + auc_val)) - (auc_val ** 2)
        se = math.sqrt((q0 + (n_pos - 1) * q1 + (n_others - 1) * q2) / (n_pos * n_others))
        auc_95_CI = 1.96 * se
        print('{} auc {:.3f} +- {:.3f}'.format(j, auc_val, auc_95_CI))

        thresh_val = show_metrics(predictions[..., i], class_indicator, thresholds)

        post_threshold = np.array(predictions[..., i] > thresh_val, np.int)

        class_filter = (ind_labels == i)

        i_name = str(i)
        out_lib[i_name + '_out'] = post_threshold[class_filter]
        out_lib[i_name + '_ids'] = ids[class_filter]
        out_lib[i_name + '_auc'] = auc_val
        out_lib[i_name + '_auc_SE'] = se

    return out_lib

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
    best_thresh = 0
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
        tn, fp, fn, tp = metrics.confusion_matrix(labels_argmax, a_thresholded).ravel()
        sens = tp / (tp + fn + eps)
        spec = tn / (tn + fp + eps)
        ppv = tp / (tp + fp + eps)
        npv = tn / (tn + fn + eps)
        acc = (tp + tn) / (tp + tn + fp + fn + eps)

        comb_val = sens + spec
        if comb_val > best_comb:
            best_comb = comb_val
            best_thresh = i
            best_thres_val = thresh_1[i]
            best_sens = sens
            best_spec = spec
            best_ppv = ppv
            best_npv = npv

        if single_man_thresh_val is not None:
            a_thresholded = np.array(predict_1 > single_man_thresh_val, np.int)
            tn, fp, fn, tp = metrics.confusion_matrix(labels_argmax, a_thresholded).ravel()
            sens = tp / (tp + fn + eps)
            spec = tn / (tn + fp + eps)
            ppv = tp / (tp + fp + eps)
            npv = tn / (tn + fn + eps)
            acc = (tp + tn) / (tp + tn + fp + fn + eps)
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

    sens_ci = 1.96 * math.sqrt((best_sens * (1 - best_sens))/ n)
    spec_ci = 1.96 * math.sqrt((best_spec * (1 - best_spec))/ n)
    ppv_ci = 1.96 * math.sqrt((best_ppv * (1 - best_ppv))/ n)
    npv_ci = 1.96 * math.sqrt((best_npv * (1 - best_npv))/ n)

    print('Best threshold val:', best_thres_val)
    print('Best threshold: {:.3f}. Sensitivity {:.3f} +- {:.3f}.  Specificity of {:.3f} +- {:.3f}.'.format(best_thresh, best_sens, sens_ci, best_spec, spec_ci))
    print('PPV {:.3f} +- {:.3f} and NPV {:.3f} +- {:.3f}'.format(best_ppv, ppv_ci, best_npv, npv_ci))
    print('\n')

    thresh_list = np.array(thresh_list)
    sens_list = np.array(sens_list)
    spec_list = np.array(spec_list)
    ppv_list = np.array(ppv_list)
    npv_list = np.array(npv_list)
    acc_list = np.array(acc_list)

    return best_thres_val
