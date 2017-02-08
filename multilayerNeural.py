import numpy as np

learningRate = 0.7
iterations = 10000
hiddenUnits = 3


def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)

    return 1 / (1 + np.exp(-x))


def backProp(target, result):
    margin = target - result


def forProp()
    hiddenSum = np.dot(inputHidden,)
