import os
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
import numpy as np
from sklearn.metrics import accuracy_score
from multiprocessing import Pool
import joblib
from math import sqrt
import argparse

def train_and_evaluate(params,train_x):
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

def train_one_class_svm(train_x, save_path):
    gammas = np.logspace(-9, 3, 13)
    nus = np.linspace(0.01, 0.99, 99)

    param_combinations = [(gamma, nu) for gamma in gammas for nu in nus]

    with Pool(processes=4) as pool:
        results = pool.starmap(train_and_evaluate, [(params, train_x) for params in param_combinations])

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

def main(train_x, save_path):
    train_one_class_svm(train_x, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_x', type=str, default='./data/train_x.npy')
    parser.add_argument('--save_path', type=str, default='./results')
    args = parser.parse_args()
    main(np.load(args.train_x),args.save_path)