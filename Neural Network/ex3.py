import sys
import numpy as np

learning_rate = 0.1
max_value = 255
file_name = "test_y"
hidden = 150
epochs = 15


def sigmoid(vec): return 1 / (1 + np.exp(-vec))


def relu(vec): return np.maximum(vec, 0)


def loss_cal(cord_y): return -1 * np.log(cord_y)


def get_txt(file): return np.loadtxt(file)


def norm(set): return np.divide(set, max_value)


def init_weights(size1, size2): return np.random.rand(size1, size2)


def save_predictions(predictions):
    file = open(file_name, 'w')
    for p in predictions:
        file.write(np.str(p)+'\n')

    file.close()


def my_shuffle(train_x, train_y):
    xy_zip = list(zip(train_x, train_y))
    np.random.shuffle(xy_zip)
    x,y = zip(*xy_zip)
    return np.asarray(x), np.asarray(y)


def init_params(input_size, hidden_size, output_size):
    w1, b1 = init_weights_uni(hidden_size, input_size), init_weights(hidden_size, 1)
    w2, b2 = init_weights_uni(output_size, hidden_size), init_weights(output_size, 1)
    params = {
      'w1': w1,
      'b1': b1,
      'w2': w2,
      'b2': b2
    }
    return params


def init_weights_uni(size1, size2):
    return np.random.uniform(-0.5, 0.5, [size1, size2])


def learn(train_x, train_y, params):
    global learning_rate
    train_x = norm(train_x)
    for e in range(epochs):
        # shuffle
        train_x, train_y = my_shuffle(train_x, train_y)
        # todo
        #sum_loss = 0
        # train
        for image, tag in zip(train_x, train_y):
            image = np.reshape(image, (1, 784))
            fp_results = fprop(image, params)
            bp_result = bprop(image, tag, fp_results)
            #sum_loss += fp_results['loss']
            params = update_params(params, bp_result)
        # sum_loss /= len(train_x)
        if e != 0:
            learning_rate /= e
    return params


def softmax(vector_x):
    exp = np.exp(vector_x)
    denominator = np.sum(exp, axis=0)
    numerator = exp
    return numerator / denominator


def d_relu(x):
    x[x < 0] = 0
    x[x > 0] = 1
    return x


def final_test(validation, params):
    predictions = []
    val_norm = norm(validation)
    for image in val_norm:
        f_dic = fprop(image, params)
        y_hat = np.argmax(f_dic['y_hat'])
        predictions.append(y_hat)
    return predictions


def fprop(image, params):
    w1, b1, w2, b2 = [params[key] for key in ('w1', 'b1', 'w2', 'b2')]
    image_t = np.reshape(image, (-1, 1))
    z1 = np.dot(w1, image_t) + b1
    h1 = relu(z1)
    if h1.max() != 0.0:
        h1 = h1 / h1.max()
    z2 = np.dot(w2, h1) + b2
    y_hat = softmax(z2)
    #loss = loss_cal(y_hat[np.int(tag)])
    return {'z1': z1, 'h1': h1, 'z2': z2, 'y_hat': y_hat, 'w1': w1, 'w2': w2}


def bprop(image, tag, params):
    z1, h1, z2, y_hat, w1, w2 = [params[key] for key in ('z1', 'h1', 'z2', 'y_hat', 'w1', 'w2')]
    y = int(tag)
    y_hat[y] -= 1
    b2_grad = y_hat
    w2_grad = np.dot(y_hat, np.transpose(h1))
    b1_grad = np.dot(np.transpose(w2), y_hat) * d_relu(z1)
    w1_grad = np.dot(b1_grad, image)
    return {'grad_w1': w1_grad, 'grad_b1': b1_grad, 'grad_w2': w2_grad, 'grad_b2': b2_grad}


def update_params(old_params, new_params):
    w1, b1, w2, b2 = [old_params[key] for key in ('w1', 'b1', 'w2', 'b2')]
    grad_w1, grad_b1, grad_w2, grad_b2 = [new_params[key] for key in ('grad_w1', 'grad_b1', 'grad_w2', 'grad_b2')]
    w1 = w1 - (learning_rate * grad_w1)
    b1 = b1 - (learning_rate * grad_b1)
    w2 = w2 - (learning_rate * grad_w2)
    b2 = b2 - (learning_rate * grad_b2)
    return {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}


def main():

    train_x = get_txt(sys.argv[1])
    train_y = get_txt(sys.argv[2])
    test_x = get_txt(sys.argv[3])
    if len(train_x) == 0 | len(train_y) == 0 | len(test_x) == 0:
        return

    input_size = train_x.shape[1]
    output_size = 10

    params = init_params(input_size, hidden, output_size)
    params = learn(train_x, train_y, params)
    predictions = final_test(test_x, params)
    save_predictions(predictions)


if __name__ == "__main__":
    main()
