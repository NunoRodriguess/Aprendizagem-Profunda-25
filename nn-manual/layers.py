#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import numpy as np
import copy

class Layer(metaclass=ABCMeta):

    @abstractmethod
    def forward_propagation(self, input):
        raise NotImplementedError
    
    @abstractmethod
    def backward_propagation(self, error):
        raise NotImplementedError
    
    @abstractmethod
    def output_shape(self):
        raise NotImplementedError
    
    @abstractmethod
    def parameters(self):
        raise NotImplementedError
    
    def set_input_shape(self, input_shape):
        self._input_shape = input_shape

    def input_shape(self):
        return self._input_shape
    
    def layer_name(self):
        return self.__class__.__name__
    

class DenseLayer (Layer):
    
    def __init__(self, n_units, input_shape = None, l1_lambda = 0.0, l2_lambda = 0.0,init_weights="he"): 
        super().__init__()
        self.n_units = n_units
        self._input_shape = input_shape
        self.init_weights = init_weights

        self.l1_lambda = l1_lambda  # Coeficiente de regularização L1
        self.l2_lambda = l2_lambda  # Coeficiente de regularização L2

        self.input = None
        self.output = None
        self.weights = None
        self.biases = None


    def initialize(self, optimizer): # init_weights = "he" ou "xavier"
        # otimizacao de pesos iniciais

        if self.init_weights == "he":
            limit = np.sqrt(2 / self.input_shape()[0])  # He initialization
            self.weights = np.random.normal(0, limit, (self.input_shape()[0], self.n_units))
        
        elif self.init_weights == "xavier":
            limit = np.sqrt(6 / (self.input_shape()[0] + self.n_units))  # Xavier/Glorot initialization
            self.weights = np.random.uniform(-limit, limit, (self.input_shape()[0], self.n_units))
        
        else:
            print("Função de ativação passada, errada!")
            raise ValueError

        self.biases = np.zeros((1, self.n_units))
        self.w_opt = copy.deepcopy(optimizer)
        self.b_opt = copy.deepcopy(optimizer)
        return self

    
    def parameters(self):
        return np.prod(self.weights.shape) + np.prod(self.biases.shape)

    def forward_propagation(self, inputs, training=True):
        self.input = inputs
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    def backward_propagation(self, output_error):
        input_error = np.dot(output_error, self.weights.T)  # dE/dX
        weights_error = np.dot(self.input.T, output_error)  # dE/dW
        bias_error = np.sum(output_error, axis=0, keepdims=True)  # dE/dB

        # regularização l1
        if self.l1_lambda > 0:
            weights_error += self.l1_lambda * np.sign(self.weights)

        # regularização L2
        if self.l2_lambda > 0: 
            weights_error += self.l2_lambda * self.weights

        # Update parameters
        self.weights = self.w_opt.update(self.weights, weights_error)
        self.biases = self.b_opt.update(self.biases, bias_error)

        return input_error
 
    def output_shape(self):
         return (self.n_units,) 
    

class DropoutLayer(Layer):
    
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self._input_shape = None # isto foi só pra nao dar erro, nao percebi o motivo disto 
        self.dropout_rate = dropout_rate
        self.mask = None
        self.input = None
        self.output = None
    
    #def initialize(self, optimizer):
    #    """
    #    nada para inicializar
    #    """
    #    return self
    
    def parameters(self):
        # nao ha parametros treinaveis, tem de ser iniciada pra nao dar erro
        return 0
    
    def forward_propagation(self, inputs, training=True):
    
        self.input = inputs
        
        # apenas aplicavel em modo treino
        if training:
            # gera uma mascara aleatoria de 0s e 1s confrome a dropout rate
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=inputs.shape)
            
            # aplica a mascara e normaliza os outputs
            self.output = inputs * self.mask / (1 - self.dropout_rate)
        
        else:
            self.output = inputs
            
        return self.output
    
    def backward_propagation(self, output_error):
        # aplica o output error à mascara da camada anterior
        return output_error * self.mask / (1 - self.dropout_rate)
    
    def output_shape(self):     
        return self.input_shape()