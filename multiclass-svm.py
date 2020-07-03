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
    if shuffle = True:
        raw_df = shuffle(raw_df)
        raw_df = raw_df.reset_index(drop=True)
    test_df = pd.read_csv(base_path + 'studentsdigits-test.csv')

    return raw_df, test_df






base_path = os.getcwd()


train_df, test_df = load_dataset(base_path)