import numpy as np
from check_grad import check_grad
from utils import *
from logistic import *
import matplotlib.pyplot as plt


def run_logistic_regression(hyperparameters, mode):
    train_inputs = []
    train_targets = []
    valid_inputs, valid_targets = load_valid()
    if mode == "normal":
        train_inputs, train_targets = load_train()
    elif mode == "small":
        train_inputs, train_targets = load_train_small()

    N, M = train_inputs.shape

    # Logistic regression weights
    # TODO:Initialize to random weights here.
    weights = np.zeros((M + 1, 1))

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    iteration_list = []
    cross_entropy_train_list = []
    cross_entropy_valid_list = []
    frac_correct_train_list = []
    frac_correct_valid_list = []
    # Begin learning with gradient descent
    for t in range(hyperparameters['num_iterations']):
        iteration_list.append(t)

        # TODO: you may need to modify this loop to create plots, etc.

        # Find the negative log likelihood and its derivatives w.r.t. the weights.
        f, df, predictions = logistic(weights, train_inputs, train_targets, hyperparameters)

        # Evaluate the prediction.
        cross_entropy_train, frac_correct_train = evaluate(train_targets, predictions)

        if np.isnan(f) or np.isinf(f):
            raise ValueError("nan/inf error")

        # update parameters
        weights = weights - hyperparameters['learning_rate'] * df / N

        # Make a prediction on the valid_inputs.
        predictions_valid = logistic_predict(weights, valid_inputs)

        # Evaluate the prediction.
        cross_entropy_valid, frac_correct_valid = evaluate(valid_targets, predictions_valid)

        # print some stats
        print(("ITERATION:{:4d}  TRAIN NLOGL:{:4.2f}  TRAIN CE:{:.6f} "
              "TRAIN FRAC:{:2.2f}  VALID CE:{:.6f}  VALID FRAC:{:2.2f}").format(
            t + 1, f / N, cross_entropy_train, frac_correct_train * 100,
            cross_entropy_valid, frac_correct_valid * 100))

        cross_entropy_train_list.append(cross_entropy_train)
        cross_entropy_valid_list.append(cross_entropy_valid)
        frac_correct_train_list.append(frac_correct_train)
        frac_correct_valid_list.append(frac_correct_valid)

    plt.title("Logistic Regression with mnist_train: Cross Entropy")
    plt.plot(iteration_list, cross_entropy_train_list, label="cross entropy train")
    plt.plot(iteration_list, cross_entropy_valid_list, label='cross entropy validation')
    plt.legend(loc="upper right")
    plt.show()

    plt.title("Logistic Regression with mnist_train: Correct Classification Rate")
    plt.plot(iteration_list, frac_correct_train_list, label="frac correct train")
    plt.plot(iteration_list, frac_correct_valid_list, label='frac correct validation')
    plt.legend(loc="upper right")
    plt.show()

    cell_text = [[cross_entropy_train_list[-1], cross_entropy_valid_list[-1]],
                  [frac_correct_train_list[-1], frac_correct_valid_list[-1]]]

    rows = ['ce', 'fac rate']
    cols = ['train set', 'validation set']

    colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
    colors = colors[::-1]

    fig, axs = plt.subplots(2, 1)
    axs[0].axis('tight')
    axs[0].axis('off')
    axs[0].table(cellText=cell_text, colLabels=cols, rowLabels=rows, loc='top', rowColours=colors)
    plt.subplots_adjust(left=0.3, top=0.8)
    plt.show()



def run_check_grad(hyperparameters):
    """Performs gradient check on logistic function.
    """

    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions + 1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,  # function to check
                      weights,
                      0.001,  # perturbation
                      data,
                      targets,
                      hyperparameters)

    print("diff =", diff)


if __name__ == '__main__':
    hyperparameters = {
        'learning_rate': 0.17,
        'weight_regularization': 1,
        'num_iterations': 46
    }

    run_logistic_regression(hyperparameters, "normal")
