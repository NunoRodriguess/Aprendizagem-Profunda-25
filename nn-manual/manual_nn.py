import numpy as np

from layers import DenseLayer, DropoutLayer
from losses import LossFunction, MeanSquaredError
from optimizer import Optimizer
from metrics import mse
from losses import BinaryCrossEntropy
from activation import SigmoidActivation, ReLUActivation
from metrics import mse, accuracy
from data import read_csv
from optimizer import Optimizer,AdamOptimizer
from neuralnet import NeuralNetwork
import numpy as np
import os
import random

def set_seed(seed: int):
    random.seed(seed) # Python
    np.random.seed(seed)  # Numpy, é o gerador utilizado pelo sklearn
    os.environ["PYTHONHASHSEED"] = str(seed)  # sistema operativo

if __name__ == '__main__':

    set_seed(25)
    # training data
    dataset_train = read_csv('train.csv', sep=',', features=True, label=True)
    dataset_test = read_csv('test.csv', sep=',', features=True, label=True)

    print("Done reading!")
    # network

    net = NeuralNetwork(epochs=20, batch_size=16, verbose=True,
                        loss=BinaryCrossEntropy, metric=accuracy, learning_rate=0.1)

    n_features = dataset_train.X.shape[1]
    net.add(DenseLayer(20, (n_features,),init_weights="xavier"))
    net.add(ReLUActivation())

    net.add(DropoutLayer(dropout_rate=0.5))

    net.add(DenseLayer(1, l1_lambda=0.01, l2_lambda=0.01,init_weights="xavier"))
    net.add(SigmoidActivation())
    #net.add(ReLUActivation())

    #net.add(DropoutLayer(droupout_rate=0.1))

    # train
    net.fit(dataset_train)

    # test
    out = net.predict(dataset_test,binary=True)
    print(f"Test: {net.score(dataset_test, out)}")
    # write predictions on file
    np.savetxt('predictions.csv', out, delimiter=',')

    # Load validation dataset
    dataset_val = read_csv('validation.csv', sep=',', features=True, label=True)

    # Get predictions
    val = net.predict(dataset_val, binary=True)

    # Get real labels
    real = dataset_val.get_y()

    # Save results with header
    output_data = np.column_stack((real, val))
    np.savetxt('validations_predictions_manual_nn.csv', output_data, delimiter=',', header="real,predicted", comments='')

    # Print validation accuracy
    print(f"Validation accuracy: {net.score(dataset_val, val)}")
