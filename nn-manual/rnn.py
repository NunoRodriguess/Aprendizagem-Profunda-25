from copy import deepcopy
from typing import Tuple
import numpy as np

from activation import SigmoidActivation,TanhActivation,ReLUActivation  # Ensure this is defined in your activation module
from layers import Layer, DenseLayer
from optimizer import AdamOptimizer,Optimizer
from losses import MeanSquaredError,BinaryCrossEntropy
from metrics import mse,accuracy,precision_recall_f1,recall,precision
from callback import EarlyStopping

class RNN(Layer):
    """A Vanilla Fully-Connected Recurrent Neural Network layer."""

    def __init__(self, n_units: int, input_shape: Tuple = None, bptt_trunc: int = 5, return_sequences: bool = False,activation=None):
        super().__init__()
        self.n_units = n_units
        self.bptt_trunc = bptt_trunc
        self._input_shape = input_shape
        self.return_sequences = return_sequences

        if activation is None:
            self.activation = TanhActivation()
        else:
            self.activation = activation

        self.W = None  # Recurrent weight
        self.V = None  # Output weight
        self.U = None  # Input weight

    def initialize(self, optimizer):
        """Initializes the weights of the layer."""
        timesteps, input_dim = self.input_shape()
        limit = 1 / np.sqrt(input_dim)
        self.U = np.random.uniform(-limit, limit, (self.n_units, input_dim))
        limit = 1 / np.sqrt(self.n_units)
        self.V = np.random.uniform(-limit, limit, (self.n_units, self.n_units))
        self.W = np.random.uniform(-limit, limit, (self.n_units, self.n_units))
        
        # Optimizers for each weight
        self.U_opt = deepcopy(optimizer)
        self.V_opt = deepcopy(optimizer)
        self.W_opt = deepcopy(optimizer)

    def forward_propagation(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        batch_size, timesteps, input_dim = inputs.shape
        self.layer_input = inputs

        self.state_input = np.zeros((batch_size, timesteps, self.n_units))
        self.states = np.zeros((batch_size, timesteps + 1, self.n_units))
        self.outputs = np.zeros((batch_size, timesteps, self.n_units))

        for t in range(timesteps):
            self.state_input[:, t] = inputs[:, t].dot(self.U.T) + self.states[:, t - 1].dot(self.W.T)
            self.states[:, t] = self.activation.activation_function(self.state_input[:, t])
            self.outputs[:, t] = self.states[:, t].dot(self.V.T)
        
        return self.outputs if self.return_sequences else self.outputs[:, -1, :]

    def backward_propagation(self, accum_grad: np.ndarray) -> np.ndarray:
        batch_size, timesteps, _ = self.layer_input.shape

        grad_U = np.zeros_like(self.U)
        grad_V = np.zeros_like(self.V)
        grad_W = np.zeros_like(self.W)
        accum_grad_next = np.zeros_like(self.layer_input)

        # Ensure correct shape when return_sequences=False
        if not self.return_sequences:
            accum_grad = np.expand_dims(accum_grad, axis=1)  # Shape (batch_size, 1, n_units)

        timesteps = accum_grad.shape[1]  # Set timesteps based on accum_grad shape

        for t in reversed(range(timesteps)):
            grad_V += accum_grad[:, t].T.dot(self.states[:, t])
            grad_wrt_state = accum_grad[:, t].dot(self.V.T) * self.activation.derivative(self.state_input[:, t])
            accum_grad_next[:, t] = grad_wrt_state.dot(self.U)
            
            for t_ in reversed(np.arange(max(0, t - self.bptt_trunc), t + 1)):
                grad_U += grad_wrt_state.T.dot(self.layer_input[:, t_])
                grad_W += grad_wrt_state.T.dot(self.states[:, t_ - 1])
                grad_wrt_state = grad_wrt_state.dot(self.W.T) * self.activation.derivative(self.state_input[:, t_ - 1])

        self.U = self.U_opt.update(self.U, grad_U)
        self.V = self.V_opt.update(self.V, grad_V)
        self.W = self.W_opt.update(self.W, grad_W)

        return accum_grad_next

    def output_shape(self) -> tuple:
        return (self._input_shape[0], self.n_units) if self.return_sequences else (self.n_units,)

    def parameters(self) -> int:
        return np.prod(self.W.shape) + np.prod(self.U.shape) + np.prod(self.V.shape)

class RecurrentNeuralNetwork:
    def __init__(self, epochs=100, batch_size=128, optimizer=None, learning_rate=0.01, momentum=0.90,
                 loss=MeanSquaredError, metric:callable = mse, callbacks=None, verbose=False):
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.loss = loss()
        self.metric = metric
        self.callbacks = callbacks if callbacks is not None else []
        if optimizer is None:
            self.optimizer = Optimizer(learning_rate=learning_rate, momentum= momentum)
        else:
            self.optimizer = optimizer
        self.layers = []
        self.history = {}
    
    def add(self, layer):
        if self.layers:
            layer.set_input_shape(self.layers[-1].output_shape())
        if hasattr(layer, 'initialize'):
            layer.initialize(self.optimizer)
        self.layers.append(layer)
    
    def forward_propagation(self, X, training=False):
        output = X
        for layer in self.layers:
            output = layer.forward_propagation(output, training=training)
        return output
    
    def backward_propagation(self, output_error):
        error = output_error
        for layer in reversed(self.layers):
            error = layer.backward_propagation(error)
        return error
    
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
   
    def fit(self, dataset):
        X = dataset.X
        y = dataset.y
        if np.ndim(y) == 1:
            y = np.expand_dims(y, axis=1)

        self.history = {}
        for epoch in range(1, self.epochs + 1):
            output_x_ = []
            y_ = []
            for X_batch, y_batch in self.get_mini_batches(X, y):
                output = self.forward_propagation(X_batch, training=True)
                error = self.loss.derivative(y_batch, output)
                self.backward_propagation(error)
                output_x_.append(output)
                y_.append(y_batch)

            output_x_all = np.concatenate(output_x_)
            y_all = np.concatenate(y_)
            loss = self.loss.loss(y_all, output_x_all)

            metric = self.metric(y_all, output_x_all) if self.metric else 'NA'
            metric_s = f"{self.metric.__name__}: {metric:.4f}" if self.metric else "NA"

            self.history[epoch] = {'loss': loss, 'metric': metric}
            if self.verbose:
                print(f"Epoch {epoch}/{self.epochs} - loss: {loss:.4f} - {metric_s}")
            
            for callback in self.callbacks:
                if hasattr(callback, 'on_epoch_end') and callback.on_epoch_end(epoch, self.history, self):
                    print(f"Training stopped at epoch {epoch}")
                    return self
        return self

    def predict(self, dataset):
        return self.forward_propagation(dataset.X)
    
    def score(self, dataset, predictions):
        if self.metric is not None:
            return self.metric(dataset.y, predictions)
        else:
            raise ValueError("No metric specified for the neural network.")


if __name__ == '__main__':
    from data import Data
    X = np.random.randn(100, 3, 3)
    y = np.random.randint(0, 2, size=(100, 1))
    dataset = Data(X, y)

    early_stopping = EarlyStopping(
        monitor='metric',  # Monitor validation metric or loss
        min_delta=0.1,       # Minimum change to qualify as improvement
        patience=1,           # Stop after 10 epochs without improvement
        verbose=True,          # Print messages
        mode='max',            # We want metric to increase (for accuracy)
        restore_best_weights=True  # Restore to best weights when stopped
    )
    
    model = RecurrentNeuralNetwork(epochs=20, batch_size=3, learning_rate=0.01, verbose=True,
                        loss=BinaryCrossEntropy, metric=accuracy, callbacks=[early_stopping])
    model.add(RNN(10, input_shape=(3, 3), bptt_trunc=5, return_sequences=True,activation=ReLUActivation()))
    model.add(RNN(5, bptt_trunc=5, return_sequences=False,activation=TanhActivation()))
    model.add(DenseLayer(1))
    model.add(SigmoidActivation())
    
    model.fit(dataset)
    predictions = model.predict(dataset)
    #print("Predictions:", predictions)
    print(f"Predictions accuracy: {model.score(dataset, predictions)}")
