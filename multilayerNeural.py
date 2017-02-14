import numpy as np
import random


class NeuralNetwork(object):
    def __init__(self, inputs, targets, iterations):
        self.inputs = inputs
        self.targets = targets
        self.iterations = iterations

        self.output_neurons = 1
        self.input_neurons = 2
        self.hidden_neurons = 5

        self.hidden_layers = 1

        self.set_count = 29

        self.weights = []
        self.hiddens = []
        self.outputs = []

        self.learning_rate = 0.7
        self.max = 0

        self.init_weights()
        self.train(self.targets)

    @staticmethod
    def sigmoid(x, derivative=False):
        if derivative:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    def forward_propagation(self, input_set):
        self.hiddens = []
        self.outputs = []

        input_set.extend([1])

        self.hiddens = [[0 for j in range(self.hidden_neurons + 1)] for i in range(self.hidden_layers)]
        self.outputs = [0 for i in range(self.output_neurons)]

        for i in range(self.hidden_neurons):
            sum = 0
            for j in range(self.input_neurons + 1):
                sum += input_set[j] * self.weights[0][j][i]

            self.hiddens[0][i] = self.sigmoid(sum)

        self.hiddens[0].extend([1])

        for i in range(self.hidden_layers - 1):
            for j in range(self.hidden_neurons):
                sum = 0
                for k in range(self.hidden_neurons + 1):
                    sum += self.hiddens[i][k] * self.weights[i + 1][k][j]

                self.hiddens[i + 1][j] = self.sigmoid(sum)

            self.hiddens[i + 1].extend([1])

        for i in range(self.output_neurons):
            sum = 0
            for j in range(self.hidden_neurons + 1):
                sum += self.hiddens[self.hidden_layers - 1][j] * self.weights[self.hidden_layers][j][i]

            self.outputs[i] = self.sigmoid(sum)

    def train_network(self, input_set, targets, print_error=False):
        delta_hiddens = [[0 for j in range(self.hidden_neurons + 1)] for i in range(self.hidden_layers)]
        delta_outputs = [0 for i in range(self.output_neurons)]

        input_set.extend([1])

        for i in range(self.hidden_layers):
            self.hiddens[i].extend([1])

        for i in range(self.output_neurons):
            error = targets[i] - self.outputs[i]
            delta_outputs[i] = self.sigmoid(self.outputs[i], True) * error

        avg_sum = 0

        for i in range(self.hidden_neurons + 1):
            error = 0
            for j in range(self.output_neurons):
                error += self.weights[self.hidden_layers][i][j] * delta_outputs[j]

            delta_hiddens[0][i] = self.sigmoid(self.hiddens[self.hidden_layers - 1][i], True) * error

            if print_error:
                avg_sum += error

        for i in range(self.hidden_layers - 1):
            for j in range(self.hidden_neurons + 1):
                error = 0
                for k in range(self.hidden_neurons):
                    error += self.weights[self.hidden_layers - 1 - i][j][k] * delta_hiddens[i][k]

                delta_hiddens[i + 1][j] = self.sigmoid(self.hiddens[self.hidden_layers - 2 - i][j], True) * error

        if print_error:
            print('Error: ', avg_sum / (self.hidden_neurons + 1))

        for i in range(self.output_neurons):
            for j in range(self.hidden_neurons + 1):
                self.weights[self.hidden_layers][j][i] += self.learning_rate * delta_outputs[i] * self.hiddens[self.hidden_layers - 1][j]

        for i in range(self.hidden_layers - 1):
            for j in range(self.hidden_neurons):
                for k in range(self.hidden_neurons + 1):
                    self.weights[self.hidden_layers - 1 - i][k][j] += self.learning_rate * delta_hiddens[i][j] * self.hiddens[self.hidden_layers - 2 - i][k]

        for i in range(self.hidden_neurons):
            for j in range(self.input_neurons + 1):
                self.weights[0][j][i] += self.learning_rate * delta_hiddens[self.hidden_layers - 1][i] * input_set[j]

    def train(self, targets):
        for i in range(self.iterations):
            for j in range(self.set_count):
                input_set = self.inputs[j]
                target_set = targets[j]

                self.forward_propagation(input_set)
                self.train_network(input_set, target_set, False)

    def init_weights(self):
        self.weights = []

        self.max = self.input_neurons
        if self.hidden_neurons > self.max:
            self.max = self.hidden_neurons

        if self.output_neurons > self.max:
            self.max = self.output_neurons

        self.max += 1

        self.weights = [[[0 for k in range(self.max)] for j in range(self.max)] for i in range(self.hidden_layers + 1)]

        for i in range(self.hidden_neurons):
            for j in range(self.input_neurons + 1):
                random_weight = random.uniform(0, 2) - 1
                self.weights[0][j][i] = random_weight

        for i in range(self.hidden_layers - 1):
            for j in range(self.hidden_neurons):
                for k in range(self.hidden_neurons + 1):
                    random_weight = random.uniform(0, 2) - 1
                    self.weights[i + 1][k][j] = random_weight

        for i in range(self.output_neurons):
            for j in range(self.hidden_neurons + 1):
                random_weight = random.uniform(0, 2) - 1
                self.weights[self.hidden_layers][j][i] = random_weight

    def solve(self, inputs, iterations=1):
        sum = []
        outsize = 0
        for i in range(iterations):
            if i != 0:
                self.weights = []
                self.hiddens = []
                self.outputs = []
                self.init_weights()
                self.train(self.targets)

            self.forward_propagation(inputs)
            output = self.outputs

            if i == 0:
                outsize = len(output)
                sum.extend([0] * outsize)

            for j in range(outsize):
                sum[j] += output[j]

        print('Result: ')
        for k in range(outsize):
            print('Output ', k + 1, ': ', sum[k] / iterations)


i = [
    [0.25, 0.14925373],
    [0.125, 0.09090909],
    [0.3333, 0.11111111],
    [0.5, 0.16666667],
    [0.2, 0.18181818],
    [0.3333, 0.13333333],
    [0.3333, 0.14492754],
    [0.3333, 0.11363636],
    [0.5, 0.20000000],
    [0.5, 0.21052632],
    [0.5, 0.06666667],
    [0.25, 0.10000000],
    [0.25, 0.11792453],
    [0.2, 0.09090909],
    [0.166666667, 0.05000000],
    [0.166666667, 0.08000000],
    [0.142857143, 0.01998002],
    [0.1, 0.04016064],
    [0.1, 0.03984064],
    [0.1, 0.03584064],
    [0.1, 0.03084064],
    [0.3333, 0.142857143],
    [0.3333, 0.138484974],
    [0.3333, 0.128379593],
    [0.3333, 0.131754042],
    [0.3333, 0.125925553],
    [0.3333, 0.117956521],
    [0.3333, 0.134852673],
    [0.3333, 0.160028165],
]

t = [
    [0],
    [0],
    [1],
    [1],
    [0],
    [1],
    [0],
    [1],
    [1],
    [0],
    [1],
    [1],
    [0],
    [0],
    [1],
    [0],
    [1],
    [0],
    [1],
    [1],
    [1],
    [0],
    [0],
    [1],
    [1],
    [1],
    [1],
    [0],
    [0],
]

iteration = 10000

for loop in range(5):
    nn = NeuralNetwork(i, t, iteration)
    print('Iterations: ', iteration)
    nn.solve(
        [0.3333, 0.833333333], 100
    )

    iteration += 10000

    print()

