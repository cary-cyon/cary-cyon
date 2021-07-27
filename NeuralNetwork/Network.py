import numpy as np


class NeuralNetwork:
    def __init__(self, inputnode, output, hiddennode, lernkf):
        # количество входных нейронов
        self.innode = inputnode
        # количестов выходных нейронов
        self.outnode = output
        # количестов нейронов на скрытом слое
        self.hnode = hiddennode
        # коэф обучения
        self.lk = lernkf
        # формирование сети
        self.WHiddenIn = np.random.rand(self.hnode, self.innode)
        self.WOutHidden = np.random.rand(self.outnode, self.hnode)
        self.activation_function = lambda x: 1 / (1 + np.exp(-x))

    # return error
    def train(self, inputs, target):
        inputs = np.array(inputs)
        target = np.array(target)
        # прямой ход
        hidden_inputs = np.dot(self.WHiddenIn, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        finall_inputs = np.dot(self.WOutHidden, hidden_outputs)
        res = self.activation_function(finall_inputs)
        # ошибка выходного слоя
        error = target - res
        # ошибка скрытого слоя
        hid_error = np.dot(self.WOutHidden.T, error)
        # изменение весов выходного слоя
        self.WOutHidden += self.lk * np.dot(error * res * (1 - res), hidden_outputs.T)
        # изменение весов скрытого слоя
        self.WHiddenIn += self.lk * np.dot(hid_error * hidden_outputs * (1 - hidden_outputs), inputs.T)
        return (sum(error ** 2))**0.5

    def query(self, inputs):
        hidden_inputs = np.dot(self.WHiddenIn, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        finall_inputs = np.dot(self.WOutHidden, hidden_outputs)
        return self.activation_function(finall_inputs)
