import io, csv, os
import numpy as np
import cvxpy as cp
import pandas as pd
from sklearn.utils import shuffle

def load(base_path):
    raw_df = pd.read_csv(base_path + 'studentspen-train.csv')
    raw_df = shuffle(raw_df)
    raw_df = raw_df.reset_index(drop=True)
    return raw_df

def get_df(i):
    temp_df = raw_df.loc[raw_df['Digit'] == i]
    temp_df = temp_df.drop(labels=['Digit'], axis='columns')
    temp_df = temp_df.reset_index(drop=True)
    return temp_df

def train_source(pos_df, neg_df, D, C=1):
    data1, data2 = pos_df.to_numpy(), neg_df.to_numpy() # 1 v 9
    N1, N2 = data1.shape[0], data2.shape[0]

    Xs = np.concatenate((data1, data2), axis=0)
    ys = np.concatenate((np.ones((N1, 1)), - np.ones((N2, 1))), axis=0)
    ws = cp.Variable((D, 1))
    bs = cp.Variable()

    epsilon = cp.Variable((N1+N2, 1))
    objective = cp.Minimize(cp.square(cp.norm(ws))*0.5 + cp.sum((epsilon)*C)/(N1+N2))
    constraints = [cp.multiply(ys, (Xs @ ws + bs)) >= 1 - epsilon,  epsilon >= 0]

    prob = cp.Problem(objective, constraints)
    prob.solve()
    print("status:", prob.status)
    print("optimal value", prob.value)
    print("optimal var ws = {}, bs = {}".format(ws.value, bs.value))

    return (ws, bs)

def hypothesis_transfer(ws, pos_df, neg_df, D, C=1):
    data1, data2 = pos_df.to_numpy(), neg_df.to_numpy() # 1 v 7
    N1, N2 = data1.shape[0], data2.shape[0]

    X = np.concatenate((data1, data2), axis=0)
    y = np.concatenate((np.ones((N1, 1)), - np.ones((N2, 1))), axis=0)

    wt = cp.Variable((D, 1))
    bt = cp.Variable()

    epsilon = cp.Variable((N1+N2, 1))
    objective = cp.Minimize(cp.square(cp.norm(wt))*0.5 + cp.sum((epsilon)*C) /(N1+N2))
    constraints = [cp.multiply(y, (X @ (ws.value + wt) + bt)) >= 1 - epsilon,  epsilon >= 0]

    prob = cp.Problem(objective, constraints)
    prob.solve()
    print("status:", prob.status)
    print("optimal value", prob.value)
    print("optimal var wt = {}, bt = {}".format(wt.value, bt.value))

    return (wt, bt)

def find_sv(pos_df, neg_df, w, b, eps=1e-8):
    data1, data2 = pos_df.to_numpy(), neg_df.to_numpy() # 1 v 9
    D, C = 8, 2
    N1, N2 = data1.shape[0], data2.shape[0]
    X = np.concatenate((data1, data2), axis=0)
    y = np.concatenate((np.ones((N1, 1)), - np.ones((N2, 1))), axis=0)

    W, B = w.value, b.value
    sup = y * (X.dot(W) + B) - 1
    idx = ((-eps<sup) & (sup<eps)).flatten()
    svX, svy = X[idx], y[idx]

    W = W.reshape(-1)
    for i in range(svX.shape[0]):
        xi = svX[i]
        print((W.dot(xi) + B), svy[i])
    
    return (svX, svy)

def instance_transfer(svX, svy, pos_df, neg_df, D, C=1):
    data1, data2 = pos_df.to_numpy(), neg_df.to_numpy() # 1 v 7
    N1, N2, N3 = data1.shape[0], data2.shape[0], svX.shape[0]

    X = np.concatenate((data1, data2), axis=0)
    y = np.concatenate((np.ones((N1, 1)), - np.ones((N2, 1))), axis=0)

    wt = cp.Variable((D, 1))
    bt = cp.Variable()

    epsilon = cp.Variable((N1+N2, 1))
    epsilon2 = cp.Variable((N3, 1))
    objective = cp.Minimize(cp.sum(cp.square(wt))*0.5 + cp.sum((epsilon)*C)/(N1+N2) + cp.sum((epsilon2)*C)/(N3))
    constraints = [cp.multiply(y, (X @ wt + bt)) >= 1 - epsilon, cp.multiply(svy, (svX @ wt + bt)) >= 1 - epsilon2,  epsilon >= 0, epsilon2 >= 0]

    prob = cp.Problem(objective, constraints)
    prob.solve()
    print("status:", prob.status)
    print("optimal value", prob.value)
    print("optimal var w = {}, b = {}".format(wt.value, bt.value))

    return (wt, bt)

def test(w, b, pos_df, neg_df):

    data1, data2 = pos_df.to_numpy(), neg_df.to_numpy()
    N1, N2 = data1.shape[0], data2.shape[0]
    X = np.concatenate((data1, data2), axis=0)
    y = np.concatenate((np.ones((N1, 1)), - np.ones((N2, 1))), axis=0)
    W, B = (w).value.reshape(-1), b.value
    count = 0
    total = N1 + N2
    errors = []

    for i in range(total):
        if y[i] * (W.dot(X[i]) + B) >= 0:
            count += 1
        else:
            errors.append((W.dot(X[i]) + B))

    return (count, total, count/total, np.mean(np.abs(np.array(errors))))

base_path = os.getcwd()
raw_df = load(base_path)

one_df = get_df(1)
one_df_orig = get_df(1)
seven_df = get_df(7)
nine_df = get_df(9)

seven_split_index = round(0.2 * len(seven_df))
seven_df_train = seven_df[ :seven_split_index]
seven_df_test = seven_df[seven_split_index: ]

one_split_index = round(0.2 * len(one_df))
one_df_train = one_df[ :one_split_index]
one_df_test = one_df[one_split_index: ]

# No transfer learning
print("Train destination problem (1 v 7) without transfer learning:")
(wn, bn) = train_source(one_df_train, seven_df_train) # 1 v 7
print("Test accuracy on destination problem:")
print(test((wn), bn, one_df_test, seven_df_test))

# Hypothesis transfer
print("\nHypothesis Transfer:")
print("Train Source Problem (1 v 9)")
(ws, bs) = train_source(one_df_orig, nine_df) # 1 v 9
print("Perform Hypothesis Transfer to destination problem (1 v 7)")
(wt, bt) = hypothesis_transfer(ws, one_df_train, seven_df_train)
print("New test accuracy on destination problem:")
print(test((ws+wt), bt, one_df_test, seven_df_test))

# Instance Transfer
print("\nInstance Transfer:")
print("Find Support Vectors from Source Problem (1 v 9):")
(svX, svy) = find_sv(one_df, nine_df, ws, bs, eps=0.2*1e-9)
print("Perform Instance Transfer to destination problem (1 v 7):")
(wi, bi) = instance_transfer(svX, svy, one_df_train, seven_df_train)
print("New test accuracy on destination problem:")
print(test(wi, bi, one_df_test, seven_df_test))




