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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = p)
    X_train_bias = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_test_bias = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    # X_bias = np.hstack([X, np.ones((X.shape[0], 1))])
    return X_train_bias, X_test_bias, y_train, y_test

def crammer_singer_classifier(X_train_bias, y_train, C=2):
    model = MultiClassClf(n_features=X_train_bias.shape[1], n_classes=10)
    
    return W, B

def test_classifier(test_df, W, B, dim):
    count = 0
    total = test_df.shape[0]
    y_test, y_pred = [], []
    for i in range(total):

        rec = test_df.iloc[i].to_numpy()
        xi = rec[:dim] / 100
        yi = rec[dim]
        pred = np.argmax(W.dot(xi) + B)
        y_test.append(yi)
        y_pred.append(pred)
        if pred == yi:
            count += 1

    return count, total, count/total

base_path = os.getcwd()


raw_df, test_df = load_dataset(base_path)

num_classes = len(set(raw_df['Digit']))
dim = len(raw_df.iloc[0]) - 1

train_df, valid_df = split_data(raw_df)

W, B = ovr_classifier(train_df, num_classes, dim)

correct_count, total, accuracy = test(valid_df, W, B, dim)