import numpy as np
from l2_distance import l2_distance
from utils import *
import matplotlib.pyplot as plt

def run_knn(k, train_data, train_labels, valid_data):
    """Uses the supplied training inputs and labels to make
    predictions for validation data using the K-nearest neighbours
    algorithm.

    Note: N_TRAIN is the number of training examples,
          N_VALID is the number of validation examples,
          and M is the number of features per example.

    Inputs:
        k:            The number of neighbours to use for classification
                      of a validation example.
        train_data:   The N_TRAIN x M array of training
                      data.
        train_labels: The N_TRAIN x 1 vector of training labels
                      corresponding to the examples in train_data
                      (must be binary).
        valid_data:   The N_VALID x M array of data to
                      predict classes for.

    Outputs:
        valid_labels: The N_VALID x 1 vector of predicted labels
                      for the validation data.
    """

    dist = l2_distance(valid_data.T, train_data.T)
    nearest = np.argsort(dist, axis=1)[:,:k]

    train_labels = train_labels.reshape(-1)
    valid_labels = train_labels[nearest]

    # note this only works for binary labels
    valid_labels = (np.mean(valid_labels, axis=1) >= 0.5).astype(np.int)
    valid_labels = valid_labels.reshape(-1,1)

    return valid_labels


if __name__ == '__main__':
    train_inputs, train_targets = load_train()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    k_values = [1, 3, 5, 7, 9]
    correct_valid = []
    correct_test = []

    for item in k_values:
        # output: predict the labels using k nearest neighbours, where (k = item).
        predict_valid = run_knn(item, train_inputs, train_targets, valid_inputs)
        predict_test = run_knn(item, train_inputs, train_targets, test_inputs)

        valid_correct_num = 0
        for i in range(len(predict_valid)):
            if predict_valid[i] == valid_targets[i]:
                valid_correct_num += 1
        correct_valid.append(valid_correct_num / len(predict_valid))

        test_correct_num = 0
        for i in range(len(predict_test)):
            if predict_test[i] == test_targets[i]:
                test_correct_num += 1
        correct_test.append(test_correct_num / len(predict_test))

    print(correct_valid)
    print(correct_test)

    plt.title("Classification Rates")
    plt.plot(k_values, correct_valid, label="Valid")
    plt.plot(k_values, correct_test, label="Test")
    plt.legend(loc="upper right")
    plt.xlabel("k values")
    plt.ylabel("classification rate")
    plt.show()

    cell_text = []
    cell_text.append(correct_valid)
    cell_text.append(correct_test)
    cell_text.reverse()

    rows = ['valid', 'test']
    colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
    colors = colors[::-1]

    fig, axs = plt.subplots(2, 1)
    axs[0].axis('tight')
    axs[0].axis('off')
    axs[0].table(cellText=cell_text, rowLabels=rows, colLabels=k_values, loc='top', rowColours=colors)
    plt.subplots_adjust(left=0.2, top=0.8)
    plt.show()
