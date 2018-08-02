import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

def update_alpha_i(alpha, gamma, x, y, i):
    return alpha[i] + gamma * (1.0 - np.sum([alpha[j] * y[i] * y[j] * pow(np.dot(x[i], x[j]) + 1.0, 2) for j in range(y.shape[0])]))

def update_alpha(alpha, gamma, x, y):
    return  [update_alpha_i(alpha, gamma, x, y, i) for i in range(y.shape[0])]

def train_svm(x, y, num_of_epoch=100, gamma=0.01):
    alpha = np.zeros(y.shape)
    for _ in range(num_of_epoch):
        alpha = update_alpha(alpha, gamma, x, y)
    return alpha

def f(alpha, x, y, X0, X1):
    return np.sum([alpha[i] * y[i] * (np.array([X0, X1]) * x[i] + 1) * (np.array([X0, X1]) * x[i] + 1) for i in range(y.shape[0])])

if __name__ == "__main__":
    x = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [1.0, 0.0],
    ])
    y = np.array([
        -1, -1,
         1,  1
    ])

    alpha = train_svm(x, y)

    xs = np.linspace(-2.0, 2.0)
    p = [[-alpha[1] + alpha[3],
          -2.0 * alpha[1] + 2.0 * alpha[3] - 2.0 * alpha[1] * X0,
          (-alpha[1] + alpha[2]) * X0 * X0 - alpha[0] - alpha[1] + alpha[2] + alpha[3]
          ] for X0 in xs]
    ys = np.array([np.roots(p[i]) for i in range(xs.shape[0])])

    plt.figure()
    plt.suptitle("Figure 2")

    positive_x = [x[i][0] for i in range(y.shape[0]) if y[i] > 0]
    positive_y = [x[i][1] for i in range(y.shape[0]) if y[i] > 0]
    negative_x = [x[i][0] for i in range(y.shape[0]) if y[i] < 0]
    negative_y = [x[i][1] for i in range(y.shape[0]) if y[i] < 0]

    plt.title("SVM alpha=" + str(alpha))
    plt.plot(positive_x, positive_y, "r*")
    plt.plot(negative_x, negative_y, "b*")
    plt.plot(xs, [i[1] if i[1] > i[0] else i[0] for i in ys], "k-")
    plt.plot(xs, [i[1] if i[1] < i[0] else i[0] for i in ys], "k-")

    plt.show()
