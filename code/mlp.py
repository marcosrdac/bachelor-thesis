# mlp.py
#
# Author: Marcos Conceicao (marcosrdac@gmail.com)
# Date: 2021-07-29

import math
from random import random
from reverse_mode import Var, exp


def rVar():
    '''Random variable generator.'''
    return Var(2 * (random() - 0.5))


def mat_vec_mul(W, x):
    '''Multiplies a matrix by a vector.'''
    res = [Var(0) for j in W]
    for i in range(len(W)):
        for j in range(len(W[0])):
            res[i] += W[i][j] * x[j]
    return res


def vec_vec_add(u, v):
    '''Adds two vectors.'''
    return [ui + vi for ui, vi in zip(u, v)]


def sigmoid(x):
    '''Sigmoid function.'''
    return Var(1) / (Var(1) + exp(-x))


def vec_sigmoid(x):
    '''Vectorized version of the sigmoid function.'''
    return [sigmoid(xi) for xi in x]


def mlp(params, x):
    '''Multilayer perceptron.'''
    for layer, (W, b) in enumerate(params):
        x = mat_vec_mul(W, x)
        x = vec_vec_add(x, b)
        not_last = layer < len(params) - 1
        if not_last:
            x = vec_sigmoid(x)
    return x


def batch_mlp(params, X):
    '''Batch version of the multilayer perceptron.'''
    return [mlp(params, xi) for xi in X]


def vec_mean(x):
    '''Takes the mean of a vector.'''
    return sum(x, start=Var(0)) / Var(len(x))


def mse(u, v):
    '''Mean squared error (MSE) function.'''
    return vec_mean([(ui - vi)**Var(2) for ui, vi in zip(u, v)])


def batch_loss(params, X, Y):
    '''Definition of the loss function from MSE.'''
    Y_pred = batch_mlp(params, X)
    losses = [mse(yi_pred, yi) for yi_pred, yi in zip(Y_pred, Y)]
    return vec_mean(losses)


def steepest_descent(params, loss_grad, step=.02):
    '''
    Defines a way to perform steepest descent on parameters given loss
    gradients and a step.
    '''
    new_params = []
    for W, b in params:
        # weights
        W_new = W.copy()
        for i in range(len(W)):
            for j in range(len(W[0])):
                # gradient descent method
                Wij_new = W[i][j].primal - step * loss_grad[W[i][j]]
                W_new[i][j] = Var(Wij_new)

        # biases
        b_new = b.copy()
        for i in range(len(b)):
            # gradient descent method
            bi_new = b[i].primal - step * loss_grad[b[i]]
            b_new[i] = Var(bi_new)

        new_params.append([W_new, b_new])
    return new_params


def mat_print(X, fmt='.2f'):
    '''Prints a matrix.'''
    for row in range(len(X)):
        start = '[[' if row == 0 else ' ['
        print(start, end='')
        for col in range(len(X[0])):
            if col < len(X[0]) - 1:
                end = '  '
            else:
                if row < len(X) - 1:
                    end = ']'
                else:
                    end = ']]'
            print(f'{X[row][col].primal:{fmt}}', end=end)
        print()


if __name__ == '__main__':
    # observations matrix
    X = [
        [Var(0), Var(0)],  # 1
        [Var(1), Var(0)],  # 2
        [Var(0), Var(1)],  # 3
        [Var(0), Var(0)],  # 4
        [Var(1), Var(1)],  # 5
        [Var(0), Var(1)],  # 6
        [Var(1), Var(0)],  # 7
        [Var(1), Var(1)],  # 8
    ]

    # targets matrix
    Y = [
        # (2x[0]+x[1], 2x[1]+x[0])
        [Var(0), Var(0)],  # 1
        [Var(2), Var(1)],  # 2
        [Var(1), Var(2)],  # 3
        [Var(0), Var(0)],  # 4
        [Var(3), Var(3)],  # 5
        [Var(1), Var(2)],  # 6
        [Var(2), Var(1)],  # 7
        [Var(3), Var(3)],  # 8
    ]

    # initial parameters for MLP architecture: 2->3->2
    # layer 1
    W1 = [  # 3x2
        [rVar(), rVar()],
        [rVar(), rVar()],
        [rVar(), rVar()],
    ]
    b1 = [  # 3
        rVar(),
        rVar(),
        rVar(),
    ]
    # layer 2
    W2 = [  # 2x3
        [rVar(), rVar(), rVar()],
        [rVar(), rVar(), rVar()],
    ]
    b2 = [  # 2
        rVar(),
        rVar(),
    ]

    # a list of parameters per layer
    params = [[W1, b1], [W2, b2]]

    print('Y (true outputs):')
    mat_print(Y)
    print()

    print('Y_pred (predictions before training):')
    Y_pred = batch_mlp(params, X)
    mat_print(Y_pred)
    print()

    # training
    epochs = 500
    print('Training with SGD...')
    for epoch in range(epochs):
        loss = batch_loss(params, X, Y)
        if epoch % 100 == 0:
            print(f'- epoch={epoch} -> loss={loss.primal:.4g}')

        loss_grad = loss.partials()
        params = steepest_descent(params, loss_grad, step=.2)
    print()
    # params is now a list of trained parameters per layer

    Y_pred = batch_mlp(params, X)
    print('Y_pred (predictions after training):')
    mat_print(Y_pred)
