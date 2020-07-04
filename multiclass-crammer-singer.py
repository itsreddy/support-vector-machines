import matplotlib.pyplot as plt 
import numpy as np
import random, os
import time
import cvxpy as cp
import pandas as pd
from sklearn.utils import shuffle
from random import randrange
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import SCORERS
from pystruct.models import MultiClassClf
from pystruct.learners import (NSlackSSVM, OneSlackSSVM,
                               SubgradientSSVM, FrankWolfeSSVM)
from time import time



def load_dataset(base_path, shuffle=True):
    raw_df = pd.read_csv(base_path + 'studentspen-train.csv')
    if shuffle == True:
        raw_df = shuffle(raw_df)
        raw_df = raw_df.reset_index(drop=True)
    test_df = pd.read_csv(base_path + 'studentsdigits-test.csv')

    return raw_df, test_df

def split_add_bias(raw_df, p=0.2):
    X = raw_df.drop(labels=['Digit'], axis='columns').to_numpy()
    y = raw_df['Digit'].to_numpy()
    if p != 0.0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = p)
        X_train_bias = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
        X_test_bias = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
        return X_train_bias, X_test_bias, y_train, y_test
    else:
        X_bias = np.hstack([X, np.ones((X.shape[0], 1))])
        return X_bias, y 

def crammer_singer_classifier(X_train_bias, y_train, num_classes, n_jobs=2, C=1):
    model = MultiClassClf(n_features=X_train_bias.shape[1], n_classes=num_classes)
    # n-slack cutting plane ssvm
    n_slack_svm = NSlackSSVM(model, n_jobs=n_jobs, verbose=0, 
                            check_constraints=False, C=C,
                            batch_size=100, tol=1e-2)

    n_slack_svm.fit(X_train_bias, y_train)
    return n_slack_svm

def test_classifier(classifier, X_test_bias, y_test):
    y_pred = np.hstack(classifier.predict(X_test_bias))
    accuracy_score = np.mean(y_pred == y_test)
    return accuracy_score, y_pred


# main

base_path = os.getcwd()
validate = True

raw_df, test_df = load_dataset(base_path)

num_classes = len(set(raw_df['Digit']))
n_dim = len(raw_df.iloc[0]) - 1

if validate == True:
    X_train_bias, X_test_bias, y_train, y_test = split_add_bias(raw_df)
    classifier = crammer_singer_classifier(X_train_bias, y_train, num_classes)
    accuracy, _ = test_classifier(classifier, X_test_bias, y_test)
    print("Accuracy of classifier on validation data: ", accuracy)
else:
    X_test_bias = np.hstack([test_df.to_numpy(), np.ones((X_test.shape[0], 1))])
    X_train_bias, y_train = split_add_bias(raw_df, p=0.0)
    classifier = crammer_singer_classifier(X_train_bias, y_train, num_classes)
    y_pred = np.hstack(classifier.predict(X_test_bias))