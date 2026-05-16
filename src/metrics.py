"""Metric helpers shared by LDNets runners."""

import numpy as np
import scipy.stats


def nrmse(prediction, reference):
    prediction = np.asarray(prediction)
    reference = np.asarray(reference)
    scale = np.max(reference) - np.min(reference)
    if scale == 0:
        return float(np.sqrt(np.mean(np.square(prediction - reference))))
    return float(np.sqrt(np.mean(np.square(prediction - reference))) / scale)


def pearson_dissimilarity(prediction, reference):
    prediction = np.reshape(np.asarray(prediction), (-1,))
    reference = np.reshape(np.asarray(reference), (-1,))
    coeff = scipy.stats.pearsonr(prediction, reference)[0]
    return float(1.0 - coeff)


def horizon_nrmse(prediction, reference):
    prediction = np.asarray(prediction)
    reference = np.asarray(reference)
    values = []
    for i_time in range(reference.shape[1]):
        values.append(nrmse(prediction[:, i_time], reference[:, i_time]))
    return [float(value) for value in values]


def parameter_count(network):
    return int(
        sum(np.prod(variable.shape.as_list()) for variable in network.trainable_variables)
    )
