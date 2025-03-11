#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from layers import DenseLayer
from losses import LossFunction, MeanSquaredError
from optimizer import Optimizer
from metrics import mse

import os
import random
def set_seed(seed: int):
    random.seed(seed) # Python
    np.random.seed(seed)  # Numpy, é o gerador utilizado pelo sklearn
    os.environ["PYTHONHASHSEED"] = str(seed)  # sistema operativo


class NeuralNetwork:
 
    def __init__(self, epochs = 1000, batch_size = 128, optimizer = None,
                 learning_rate = 0.01, momentum = 0.90, verbose = False, 
                 loss = MeanSquaredError,
                 metric:callable = mse):
        self.epochs = epochs
        self.batch_size = batch_size
        if optimizer is None:
            self.optimizer = Optimizer(learning_rate=learning_rate, momentum= momentum)
        else:
            self.optimizer = optimizer
        self.verbose = verbose
        self.loss = loss()
        self.metric = metric

        # attributes
        self.layers = []
        self.history = {}

    def add(self, layer):
        if self.layers:
            layer.set_input_shape(input_shape=self.layers[-1].output_shape())
        if hasattr(layer, 'initialize'):
            layer.initialize(self.optimizer)
        self.layers.append(layer)
        return self

    def get_mini_batches(self, X, y = None,shuffle = True):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        assert self.batch_size <= n_samples, "Batch size cannot be greater than the number of samples"
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, n_samples - self.batch_size + 1, self.batch_size):
            if y is not None:
                yield X[indices[start:start + self.batch_size]], y[indices[start:start + self.batch_size]]
            else:
                yield X[indices[start:start + self.batch_size]], None

    def forward_propagation(self, X, training):
        output = X
        for layer in self.layers:
            output = layer.forward_propagation(output, training)
        return output

    def backward_propagation(self, output_error):
        error = output_error
        for layer in reversed(self.layers):
            error = layer.backward_propagation(error)
        return error

    def fit(self, dataset):
        X = dataset.X
        y = dataset.y
        if np.ndim(y) == 1:
            y = np.expand_dims(y, axis=1)

        self.history = {}
        for epoch in range(1, self.epochs + 1):
            # store mini-batch data for epoch loss and quality metrics calculation
            output_x_ = []
            y_ = []
            for X_batch, y_batch in self.get_mini_batches(X, y):
                # Forward propagation
                output = self.forward_propagation(X_batch, training=True)
                # Backward propagation
                error = self.loss.derivative(y_batch, output)
                self.backward_propagation(error)

                output_x_.append(output)
                y_.append(y_batch)

            output_x_all = np.concatenate(output_x_)
            y_all = np.concatenate(y_)

            # compute loss
            loss = self.loss.loss(y_all, output_x_all)

            if self.metric is not None:
                metric = self.metric(y_all, output_x_all)
                metric_s = f"{self.metric.__name__}: {metric:.4f}"
            else:
                metric_s = "NA"
                metric = 'NA'

            # save loss and metric for each epoch
            self.history[epoch] = {'loss': loss, 'metric': metric}

            if self.verbose:
                print(f"Epoch {epoch}/{self.epochs} - loss: {loss:.4f} - {metric_s}")

        return self

    def predict(self, dataset, binary=False):
        out = self.forward_propagation(dataset.X, training=False)
        if binary:
            return np.where(out >= 0.5, 1, 0)
        else:
            return out

    def score(self, dataset, predictions):
        if self.metric is not None:
            return self.metric(dataset.y, predictions)
        else:
            raise ValueError("No metric specified for the neural network.")


if __name__ == '__main__':
    from losses import BinaryCrossEntropy
    from activation import SigmoidActivation, ReLUActivation
    from metrics import mse, accuracy
    from data import read_csv
    from optimizer import Optimizer, AdamOptimizer

    set_seed(25)

    # Carregar os dados
    dataset_train = read_csv('train.csv', sep=',', features=True, label=True)
    dataset_test = read_csv('test.csv', sep=',', features=True, label=True)
    dataset_stor = read_csv('../reg_logistica/input_prof.csv', sep=',', features=True, label=False)

    print("Done reading!")

    # Verificar o balanceamento das classes no treino
    train_pos = np.sum(dataset_train.y)
    train_neg = len(dataset_train.y) - train_pos
    print(f"Distribuição no treino: Positivos={train_pos}, Negativos={train_neg}, Ratio={train_pos/len(dataset_train.y):.2f}")

    # Criar e treinar a rede neural
    net = NeuralNetwork(epochs=20, batch_size=16, verbose=True,
                        loss=BinaryCrossEntropy, metric=accuracy, learning_rate=0.1)
    n_features = dataset_train.X.shape[1]
    net.add(DenseLayer(20, (n_features,)))
    net.add(ReLUActivation())
    net.add(DenseLayer(1))
    net.add(SigmoidActivation())

    # Treinar o modelo
    net.fit(dataset_train)

    # Testar o modelo no conjunto de teste
    out_test = net.predict(dataset_test, binary=True)
    test_accuracy = net.score(dataset_test, out_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Validar o modelo no conjunto de validação
    dataset_val = read_csv('validation.csv', sep=',', features=True, label=True)
    out_val = net.predict(dataset_val, binary=True)
    val_accuracy = net.score(dataset_val, out_val)
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    # Aplicar o modelo ao dataset input_stor
    out_stor = net.predict(dataset_stor, binary=True)

    # Converter previsões para labels (Human ou AI)
    binary_conv = {0: "Human", 1: "AI"}
    out_labels = np.vectorize(binary_conv.get)(out_stor)

    # Criar IDs para as previsões
    num_samples = len(out_labels)
    ids = [f"D1-{i+1}" for i in range(num_samples)]

    # Empilhar IDs e labels
    output_array = np.column_stack((ids, out_labels))

    # Salvar as previsões no formato desejado
    np.savetxt('predictions_stor.csv', output_array, delimiter='\t', fmt='%s', header="ID\tLabel", comments='')

    # Verificar a semelhança entre os dados de treino e input_stor
    def compare_distributions(train_data, stor_data):
        print("\nComparando distribuições:")
        print(f"Média (Treino): {np.mean(train_data, axis=0)}")
        print(f"Média (Input_stor): {np.mean(stor_data, axis=0)}")
        print(f"Desvio Padrão (Treino): {np.std(train_data, axis=0)}")
        print(f"Desvio Padrão (Input_stor): {np.std(stor_data, axis=0)}")