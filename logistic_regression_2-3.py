from check_grad import check_grad
from utils import *
from logistic import *
import matplotlib.pyplot as plt


def run_logistic_regression(hyperparameters, mode, runs):
    print(hyperparameters['weight_regularization'])
    train_inputs = []
    train_targets = []
    valid_inputs, valid_targets = load_valid()
    if mode == "normal":
        train_inputs, train_targets = load_train()
    elif mode == "small":
        train_inputs, train_targets = load_train_small()

    N, M = train_inputs.shape

    sum_cross_entropy_train = 0
    sum_cross_entropy_valid = 0.0
    sum_classification_rate_train = 0.0
    sum_clasification_rate_valid = 0.0

    for i in range(runs):

        # Logistic regression weights
        # TODO:Initialize to random weights here.
        # weights = np.transpose([np.random.normal(0, 0.1, M + 1)])
        weights = np.zeros((M + 1, 1))

        # Verify that your logistic function produces the right gradient.
        # diff should be very close to 0.
        run_check_grad(hyperparameters)

        # Begin learning with gradient descent
        cross_entropy_train_last = 0
        cross_entropy_valid_last = 0
        frac_correct_train_last = 0
        frac_correct_valid_last = 0
        for t in range(hyperparameters['num_iterations']):

            # TODO: you may need to modify this loop to create plots, etc.

            # Find the negative log likelihood and its derivatives w.r.t. the weights.
            f, df, predictions = logistic_pen(weights, train_inputs, train_targets, hyperparameters)

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

            cross_entropy_train_last = cross_entropy_train
            cross_entropy_valid_last = cross_entropy_valid
            frac_correct_train_last = frac_correct_train
            frac_correct_valid_last = frac_correct_valid

        sum_cross_entropy_train += cross_entropy_train_last
        sum_cross_entropy_valid += cross_entropy_valid_last
        sum_classification_rate_train += frac_correct_train_last
        sum_clasification_rate_valid += frac_correct_valid_last

    result = [sum_cross_entropy_train / runs, sum_cross_entropy_valid / runs, sum_classification_rate_train / runs,
              sum_clasification_rate_valid / runs]
    return result



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
        'learning_rate': 0.2,
        'weight_regularization': 1,
        'num_iterations': 55
    }

    regularzation = [0, 0.001, 0.01, 0.1, 1.0]

    average_cross_entropy_train_list = []
    average_cross_entropy_valid_list = []
    average_error_train_list = []
    average_error_valid_list = []

    for reg in regularzation:
        hyperparameters['weight_regularization'] = reg
        result = run_logistic_regression(hyperparameters, "small", 10)
        average_cross_entropy_train_list.append(result[0])
        average_cross_entropy_valid_list.append(result[1])
        average_error_train_list.append(100-result[2] * 100)
        average_error_valid_list.append(100-result[3] * 100)

    xaxis = np.arange(0, 1, step=0.2)
    plt.title("Penalized Logistic Regression with mnist_train: Cross Entropy")
    plt.xticks(xaxis, ('0', '0.001', '0.01', '0.1', '1'))
    plt.plot(xaxis, average_cross_entropy_train_list, label='cross entropy train')
    plt.plot(xaxis, average_cross_entropy_valid_list, label='cross entropy valid')
    plt.axis([0, 1, 0, 50])
    plt.xlabel('regular')
    plt.ylabel('average cross entropy')
    plt.legend(loc="upper right")
    plt.show()

    cell_text = [average_cross_entropy_train_list, average_cross_entropy_valid_list]

    cols = ['0', '0.001', '0.01', '0.1', '1']
    rows = ['train set', 'validation set']

    colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
    colors = colors[::-1]

    fig, axs = plt.subplots(2, 1)
    axs[0].axis('tight')
    axs[0].axis('off')
    axs[0].table(cellText=cell_text, colLabels=cols, rowLabels=rows, loc='top', rowColours=colors)
    plt.subplots_adjust(left=0.3, top=0.8)
    plt.show()

    xaxis = np.arange(0, 1, step=0.2)
    plt.title("Penalized Logistic Regression with mnist_train: Error")
    plt.xticks(xaxis, ('0', '0.001', '0.01', '0.1', '1'))
    plt.plot(xaxis, average_error_train_list, label='cross entropy train')
    plt.plot(xaxis, average_error_valid_list, label='cross entropy valid')
    plt.axis([0, 1, 0, 50])
    plt.xlabel('regular')
    plt.ylabel('average cross entropy')
    plt.legend(loc="upper right")
    plt.show()

    cell_text = [average_error_train_list, average_error_valid_list]
    fig1, axs1 = plt.subplots(2, 1)
    cols = ['0', '0.001', '0.01', '0.1', '1']
    rows = ['train set', 'validation set']
    colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
    colors = colors[::-1]
    axs1[0].axis('tight')
    axs1[0].axis('off')
    axs1[0].table(cellText=cell_text, colLabels=cols, rowLabels=rows, loc='top', rowColours=colors)
    plt.subplots_adjust(left=0.3, top=0.8)
    plt.show()
