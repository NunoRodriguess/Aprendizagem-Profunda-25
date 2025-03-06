#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from layers import DenseLayer, DropoutLayer
from losses import LossFunction, MeanSquaredError
from optimizer import Optimizer
from metrics import mse


class NeuralNetwork:
 
    def __init__(self, epochs = 100, batch_size = 128, optimizer = None,
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

    def predict(self, dataset):
        return self.forward_propagation(dataset.X, training=False)

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
    from optimizer import Optimizer,AdamOptimizer

    # training data
    dataset_train = read_csv('../train.csv', sep=',', features=True, label=True)
    dataset_test = read_csv('../test.csv', sep=',', features=True, label=True)

    print("Done reading!")
    # network
    net = NeuralNetwork(epochs=10, batch_size=30, verbose=True,
                        loss=BinaryCrossEntropy, metric=accuracy, optimizer=AdamOptimizer(learning_rate=0.01))
    n_features = dataset_train.X.shape[1]
    net.add(DenseLayer(6, (n_features,)))
    net.add(ReLUActivation())

    net.add(DropoutLayer(dropout_rate=0.5))

    net.add(DenseLayer(1, l1_lambda=0.01, l2_lambda=0.01))
    net.add(SigmoidActivation())
    #net.add(ReLUActivation())

    #net.add(DropoutLayer(droupout_rate=0.1))

    # train
    net.fit(dataset_train)

    # test
    out = net.predict(dataset_test)
    print(f"Test: {net.score(dataset_test, out)}")
    # write predictions on file
    np.savetxt('../predictions.csv', out, delimiter=',')

    # validation
    dataset_val = read_csv('../validation.csv', sep=',', features=True, label=True)
    val = net.predict(dataset_val)
    print(net.score(dataset_val, val))
    print(f"Validation: {net.score(dataset_val, val)}")
