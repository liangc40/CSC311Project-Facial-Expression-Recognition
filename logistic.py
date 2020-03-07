""" Methods for doing logistic regression."""

import numpy as np
from utils import sigmoid

def logistic_predict(weights, data):
    """
    Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples and
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to the bias (intercepts).
        data:       N x M data matrix where each row corresponds
                    to one data point.
    Outputs:
        y:          :N x 1 vector of probabilities. This is the output of the classifier.
    """
    augmented_data = np.ones((data.shape[0], data.shape[1] + 1))
    augmented_data[:, :-1] = data
    z = np.dot(augmented_data, weights)
    return sigmoid(z)


def evaluate(targets, y):
    """
    Compute evaluation metrics.
    Inputs:
        targets : N x 1 vector of targets.
        y       : N x 1 vector of probabilities.
    Outputs:
        ce           : (scalar) Cross entropy. CE(p, q) = E_p[-log q]. Here we want to compute CE(targets, y)
        frac_correct : (scalar) Fraction of inputs classified correctly.
    """
    # ce = -(np.sum(np.dot(np.transpose(1 - targets), np.log(1.0 - y))+np.dot(np.transpose(targets), np.log(y))))
    ce = -np.multiply(targets, np.log(y)) - np.multiply(1-targets, np.log(1-y))
    counter = 0
    for i in range(targets.shape[0]):
        if y[i] >= 0.5 and targets[i] == 1:
            counter += 1
        elif y[i] < 0.5 and targets[i] == 0:
            counter += 1
    frac_correct = counter / len(targets)
    return np.sum(ce), frac_correct


def logistic(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:       The sum of the loss over all data points. This is the objective that we want to minimize.
        df:      (M+1) x 1 vector of derivative of f w.r.t. weights.
        y:       N x 1 vector of probabilities.
    """
    y = logistic_predict(weights, data)
    f = evaluate(targets, y)[0]
    augmented_data = np.ones((data.shape[0], data.shape[1] + 1))
    augmented_data[:, :-1] = data
    df = np.dot(np.transpose(augmented_data), y - targets)

    return f, df, y


def logistic_pen(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:             The sum of the loss over all data points. This is the objective that we want to minimize.
        df:            (M+1) x 1 vector of derivative of f w.r.t. weights.
    """
    c = np.append(np.ones((data.shape[1], 1)), [[0]], axis=0)
    f, df, y = logistic(weights, data, targets, hyperparameters)
    f = f + sum(((weights * c) ** 2) * hyperparameters['weight_regularization'] / 2)
    df = df + weights * c * hyperparameters['weight_regularization']
    return f, df, y
