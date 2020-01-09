import numpy as np
import sys
from scipy import stats
import numpy.core.defchararray as npy_dy


def min_max_normalization(train_x):
    for i in range(0, train_x.shape[1]):
        train_x[:, i] = ((train_x[:, i] - train_x[:, i].min()) / (train_x[:, i].max() - train_x[:, i].min()))
    return train_x


def z_score_normalization(train_x):
    for i in range(0, train_x.shape[1]):
        vector = train_x[:, i]
        train_x[:, i] = stats.mstats.zscore(vector)
    return train_x


def my_shuffle(arr_train_x, arr_train_y):
    xy_zip = list(zip(arr_train_x, arr_train_y))
    np.random.shuffle(xy_zip)
    return xy_zip


def predict_yhat(w, x):
    return np.argmax(np.dot(w, x))


def svm(train_x, train_y, w, epochs):
    lamda = 0.01
    eta = 0.1
    for e in range(epochs):
        xy_zip = my_shuffle(train_x, train_y)
        train_x, train_y = zip(*xy_zip)
        for x, y in zip(train_x, train_y):
            y_hat = predict_yhat(w, x)
            # update
            if y != y_hat:
                for i in range(w.shape[0]):
                    if i == y:
                        w[y, :] = ((1 - (eta * lamda)) * w[y, :]) + (eta * x)
                    elif i == y_hat:
                        w[y_hat, :] = ((1 - (eta * lamda)) * w[y_hat, :]) - (eta * x)
                    else:
                        w[i, :] = w[i, :] * (1 - (eta * lamda))
            else:
                for i in range(w.shape[0]):
                    w[i, :] = w[i, :] * (1 - eta * lamda)
        if e > 0:
            eta = eta / e
            lamda = lamda / 100
    return w


def pa(train_x, train_y, w, epochs):
    new_w = w
    count = 0
    for e in range(epochs):
        xy_zip = my_shuffle(train_x, train_y)
        train_x, train_y = zip(*xy_zip)
        for x, y in zip(train_x, train_y):
            y_hat = predict_yhat(w, x)
            tau = cal_tau(x, y, y_hat, w)
            # update
            if y != y_hat:
                w[y, :] = w[y, :] + (tau * x)
                w[y_hat, :] = w[y_hat, :] - (tau * x)
                new_w = new_w + w
                count += 1
    if count != 0:
        w = new_w / count
    return w


def perceptron(train_x, train_y, w_arr, epochs):
    eta = 0.01
    for e in range(epochs):
        xy_zip = my_shuffle(train_x, train_y)
        train_x, train_y = zip(*xy_zip)
        for x, y in zip(train_x, train_y):
            y_hat = predict_yhat(w_arr, x)
            # update
            if y != y_hat:
                w_arr[y, :] = w_arr[y, :] + (eta * x)
                w_arr[y_hat, :] = w_arr[y_hat, :] - (eta * x)
        if e != 0:
            eta = eta / e
    return w_arr


def learn(algorithm, epochs, train_x, train_y):
    w_arr = np.zeros((3, train_x.shape[1]))
    # run the algorithm
    w_arr = algorithm(train_x, train_y, w_arr, epochs)
    return w_arr


def cal_tau(x, y, y_hat, w_arr):
    loss = max(0.0, 1 - np.dot(w_arr[y], x) + np.dot(w_arr[y_hat], x))
    tau = loss / ((np.power(np.linalg.norm(x, ord=2), 2)) * 2)
    return tau


def get_x(file):
    # read the train-x file
    train_x = np.genfromtxt(file, dtype='str', delimiter=',')
    train_x = npy_dy.replace(train_x, 'M', '0.25')
    train_x = npy_dy.replace(train_x, 'F', '0.5')
    train_x = npy_dy.replace(train_x, 'I', '0.75')
    return train_x.astype(np.float)


def get_y(file):
    # read the train-y file
    train_y = np.genfromtxt(file, dtype='str', delimiter='\n')
    train_y = train_y.astype(np.float)
    return train_y.astype(np.int)


def predictions(test_x, w):
    results = []
    m = len(test_x)
    for i in range(m):
        # predict
        y_hat = predict_yhat(w, test_x[i])
        results.append(y_hat)
    return results


def main():
    # get from the command line
    train_x = get_x(sys.argv[1])
    train_y = get_y(sys.argv[2])
    validation = get_x(sys.argv[3])

    # learn from the training set and test the validation set
    w = learn(perceptron, 18, train_x, train_y)
    pre_predictions = predictions(validation, w)

    w = learn(pa, 15, train_x, train_y)
    pa_predictions = predictions(validation, w)

    w = learn(svm, 25, train_x, train_y)
    svm_predictions = predictions(validation, w)

    # printing each algorithm's prediction of the validation set.
    for i in range(len(pre_predictions)):
        print("perceptron:", repr(pre_predictions[i]) + ',', "svm:", repr(svm_predictions[i]) + ',', "pa:",
              repr(pa_predictions[i]))


if __name__ == "__main__":
    main()
