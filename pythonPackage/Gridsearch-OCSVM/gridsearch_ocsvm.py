import os
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
import numpy as np
from sklearn.metrics import accuracy_score
from multiprocessing import Pool
import joblib
from math import sqrt
import argparse

class gridsearch_ocsvm:
    def __init__(self,gamma = np.logspace(-9, 3, 13), nu = np.linspace(0.01, 0.99, 99)):
        '''
        Initialize the parameters for gridsearch
        default gamma = np.logspace(-9, 3, 13)
        default nu = np.linspace(0.01, 0.99, 99)
        '''
        self.gamma = gamma
        self.nu = nu
        
    def train_one_class_svm(self,train_x, save_path):
        '''
        Inputs: train_x - numpy array of training data
                save_path - path to save the trained model

        This function is used to train the model using the gridsearch method. 
        The model is trained using the training data and the best parameters are selected based on the highest accuracy then later chooses the highest score if there are multiple parameters with the same accuracy.
        The model is saved at the save_path.
        Pooling is used to parallelize the training process.
        
        Returns: best_params - tuple of the best parameters (gamma, nu)
                 best_accuracy - float of the best accuracy          
        '''

        param_combinations = [(gamma, nu) for gamma in self.gamma for nu in self.nu]

        with Pool(processes=4) as pool:
            results = pool.starmap(self._train_and_evaluate, [(params, train_x) for params in param_combinations])

        results = [result for result in results if result is not None]
        highest_accuracy = max(results, key=lambda x: x[2])[2]
        highest_accuracy_tuples = [t for t in results if t[2] >= highest_accuracy]


        best_gamma, best_nu, best_accuracy, score = max(highest_accuracy_tuples, key=lambda x: x[3])
        best_params = (best_gamma, best_nu)
        print("Best Parameters (Gamma, Nu):", best_params)
        print("Best Accuracy:", best_accuracy)

        best_gamma, best_nu = best_params
        best_clf = OneClassSVM(gamma=best_gamma, nu=best_nu)
        best_clf.fit(train_x)

        save_path = os.path.join(save_path, "svm_model.pkl")
        joblib.dump(best_clf, save_path)
        print("Trained SVM model saved at:", save_path)

        return best_params, best_accuracy
    
    def _train_and_evaluate(self,params,train_x):
        '''
        This function is used to train and evaluate the model for a given a parameter pair and give an output.
        Score is calculated using the formula ( gamma + nu ) - abs( gamma - nu ) to give more weight to the difference between gamma and nu.

        Returns the accuracy and score for each parameter combination if there is atleast one positive prediction (i.e. accuracy is not 0).
        '''
        gamma, nu = params

        clf = OneClassSVM(gamma=gamma, nu=nu)
        clf.fit(train_x)
        y_pred = clf.predict(train_x)

        if 1. in y_pred:
            accuracy = accuracy_score(np.ones(len(y_pred)), y_pred)
            score = (gamma+nu)-abs(gamma-nu)
            return (gamma, nu, accuracy, score)
        else:
            return None