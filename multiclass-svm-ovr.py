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

def split_data(raw_df, p=0.2):
    split_index = round((1-p)* len(raw_df))
    train_df = raw_df[ :split_index]
    valid_df = raw_df[split_index: ]
    return train_df, valid_df

def create_dfs(train_df, num_classes):
    dfs = []
    for i in range(num_classes):
        temp_df = train_df.loc[raw_df['Digit'] == i]
        temp_df = temp_df.drop(labels=['Digit'], axis='columns')
        temp_df = temp_df.reset_index(drop=True)
        dfs.append(temp_df)
    return dfs

def ovr_classifier(train_df, num_classes, dim, C=2):
    weights, bs = [], []
    dfs = create_dfs(train_df, num_classes)
    for i in range(num_classes):
        data1 = dfs[i].to_numpy()
        rest = []
        for j in range(num_classes):
            if j != i:
                rest.append(dfs[j])
        restdf = pd.concat(rest, ignore_index=True, sort=False)
        data2 = restdf.to_numpy()

        D = dim
        N1 = data1.shape[0]
        N2 = data2.shape[0]

        data1 = data1 / 100
        data2 = data2 / 100

        X = np.concatenate((data1, data2), axis=0)
        y = np.concatenate((np.ones((N1, 1)), - np.ones((N2, 1))), axis=0)

        w = cp.Variable((D, 1))
        b = cp.Variable()
        epsilon = cp.Variable((N1+N2, 1))
        objective = cp.Minimize(cp.sum(cp.square(w))*0.5 + cp.sum(cp.square(epsilon)*C))
        constraints = [cp.multiply(y, (X @ w + b)) >= 1 - epsilon,  epsilon >= 0]

        prob = cp.Problem(objective, constraints)
        prob.solve()

        weights.append(w.value.reshape(-1))
        bs.append(b.value)


base_path = os.getcwd()


raw_df, test_df = load_dataset(base_path)

num_classes = len(set(raw_df['Digit']))
dim = len(raw_df.iloc[0]) - 1

train_df, valid_df = split_data(raw_df)