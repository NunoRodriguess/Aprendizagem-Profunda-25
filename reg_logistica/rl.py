# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


import os
import random
def set_seed(seed: int):
    random.seed(seed) # Python
    np.random.seed(seed)  # Numpy, é o gerador utilizado pelo sklearn
    os.environ["PYTHONHASHSEED"] = str(seed)  # sistema operativo



class LogisticRegression:
    
    def __init__(self, dataset, standardize = False, regularization = False, lamda = 1):
        if standardize:
            dataset.standardize()
            self.X = np.hstack ((np.ones([dataset.nrows(),1]), dataset.Xst ))
            self.standardized = True
        else:
            self.X = np.hstack ((np.ones([dataset.nrows(),1]), dataset.X ))
            self.standardized = False
        self.y = dataset.y
        self.theta = self.theta = np.zeros(self.X.shape[1])
        self.regularization = regularization
        self.lamda = lamda
        self.data = dataset

    def buildModel(self):
        if self.regularization:
            self.optim_model_reg(self.lamda)    
        else:
            self.optim_model()

    def gradientDescent(self, alpha = 0.01, iters = 10000):
        m = self.X.shape[0]
        n = self.X.shape[1]
        self.theta = np.zeros(n)  
        for its in range(iters):
            J = self.costFunction()
            if its%1000 == 0: print(J)
            delta = self.X.T.dot(sigmoid(self.X.dot(self.theta)) - self.y)                      
            self.theta -= (alpha /m  * delta )    
    
    def optim_model(self):
        from scipy import optimize

        n = self.X.shape[1]
        options = {'full_output': True, 'maxiter': 500}
        initial_theta = np.zeros(n)
        self.theta, _, _, _, _ = optimize.fmin(lambda theta: self.costFunction(theta), initial_theta, **options)
    
    
    def optim_model_reg(self, lamda):
        from scipy import optimize

        n = self.X.shape[1]
        initial_theta = np.ones(n)        
        result = optimize.minimize(lambda theta: self.costFunctionReg(theta, lamda), initial_theta, method='BFGS', 
                                    options={"maxiter":500, "disp":False} )
        self.theta = result.x    
  
    
    def predict(self, instance):
        p = self.probability(instance)
        if p >= 0.5: res = 1
        else: res = 0
        return res
    
    def probability(self, instance):
        x = np.empty([self.X.shape[1]])        
        x[0] = 1
        x[1:] = np.array(instance[:self.X.shape[1]-1])
        if self.standardized:
            if np.all(self.sigma!= 0): 
                x[1:] = (x[1:] - self.data.mu) / self.data.sigma
            else: x[1:] = (x[1:] - self.mu) 
        
        return sigmoid ( np.dot(self.theta, x) )


    def costFunction(self, theta = None):
        if theta is None: theta= self.theta        
        m = self.X.shape[0]
        p = sigmoid ( np.dot(self.X, theta) )
        cost  = (-self.y * np.log(p) - (1-self.y) * np.log(1-p) )
        res = np.sum(cost) / m
        return res
        
    def costFunctionReg(self, theta = None, lamda = 1):
        if theta is None: theta= self.theta        
        m = self.X.shape[0]
        p = sigmoid ( np.dot(self.X, theta) )
        cost  = (-self.y * np.log(p) - (1-self.y) * np.log(1-p) )
        reg = np.dot(theta[1:], theta[1:]) * lamda / (2*m)
        return (np.sum(cost) / m) + reg
        
    def printCoefs(self):
        print(self.theta)

    def predictMany(self, Xt):
        p = sigmoid ( np.dot(Xt, self.theta) )
        return np.where(p >= 0.5, 1, 0)
    
    def accuracy(self, Xt, yt):
        preds = self.predictMany(Xt)
        errors = np.abs(preds-yt)
        return 1.0 - np.sum(errors)/yt.shape[0]
    

    def score(self, dataset):
        Xt = np.hstack((np.ones([dataset.nrows(), 1]), dataset.X))
        yt = dataset.y
        return self.accuracy(Xt, yt)
    
def sigmoid(x):
  return 1 / (1 + np.exp(-x))
  
def mapFeature(X1, X2, degrees = 6):
	out = np.ones( (np.shape(X1)[0], 1) )
	
	for i in range(1, degrees+1):
		for j in range(0, i+1):
			term1 = X1 ** (i-j)
			term2 = X2 ** (j)
			term  = (term1 * term2).reshape( np.shape(term1)[0], 1 ) 
			out   = np.hstack(( out, term ))
	return out  
  
  
if __name__ == '__main__':
    from data import read_csv 

    set_seed(25)

    # Carregar os dados
    dataset_train = read_csv('train.csv', sep=',', features=True, label=True)
    dataset_test = read_csv('test.csv', sep=',', features=True, label=True)
    dataset_val = read_csv('validation.csv', sep=',', features=True, label=True)

    print("Done reading!")

    # Criar e treinar o modelo
    log_model = LogisticRegression(dataset_train, standardize=True, regularization=True, lamda=0.1)
    log_model.buildModel()  

    # Testar o modelo
    test_predictions = log_model.predictMany(np.hstack((np.ones([dataset_test.nrows(), 1]), dataset_test.X)))
    test_accuracy = log_model.score(dataset_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Validar o modelo
    val_predictions = log_model.predictMany(np.hstack((np.ones([dataset_val.nrows(), 1]), dataset_val.X)))
    val_accuracy = log_model.score(dataset_val)
    print(f"Validation Accuracy: {val_accuracy:.4f}")