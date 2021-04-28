
import numpy as np
from MMFA import MMFA
import pandas as pd
import os 
import matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import FastICA, PCA
from matplotlib.ticker import MultipleLocator
from numpy import genfromtxt
from utils import softmax

dirname = os.path.dirname(__file__)

filename = os.path.join(dirname, "yellow_tripdata_2019-02_post.csv")
data_raw = pd.read_csv(filename)

data_raw = data_raw.sample(frac=0.005)

X = pd.concat([data_raw['passenger_count'],
                   data_raw['fare_amount'],
                   data_raw['tip_amount']], axis = 1).to_numpy()
X = (X - X.mean(axis = 0)) / X.std(axis = 0)
pca = PCA(n_components=3, whiten=True)
X = pca.fit_transform(X)

Du = 3

Y = pd.concat([
               pd.get_dummies(data_raw['PULocationID'], drop_first=True),
               pd.get_dummies(data_raw['hour'], drop_first=True),
               pd.get_dummies(data_raw['dayofweek'], drop_first=True)], axis = 1).to_numpy()
Y_ext = pd.concat([
               pd.get_dummies(data_raw['PULocationID'], drop_first=False),
               pd.get_dummies(data_raw['hour'], drop_first=False),
               pd.get_dummies(data_raw['dayofweek'], drop_first=False)], axis = 1).to_numpy()

Mu = np.array([
      pd.get_dummies(data_raw['PULocationID'], drop_first=True).shape[1],
      pd.get_dummies(data_raw['hour'], drop_first=True).shape[1],
      pd.get_dummies(data_raw['dayofweek'], drop_first=True).shape[1]])
Mu_ext = Mu + 1
I = len(X)


LL_MMFA_vec = []
Perp_MMFA_vec = []

for exp in range(0, 10):

    ind = np.arange(0, len(data_raw))
    np.random.shuffle(ind)
    
    X_train = X[ind[0:int(len(data_raw)*0.6)]]
    X_val = X[ind[int(len(data_raw)*0.6):int(len(data_raw)*0.8)]]
    X_test = X[ind[int(len(data_raw)*0.8):]]
    
    Y_train = Y[ind[0:int(len(data_raw)*0.6)]]
    Y_val = Y[ind[int(len(data_raw)*0.6):int(len(data_raw)*0.8)]]
    Y_test = Y[ind[int(len(data_raw)*0.8):]]
    
    Y_train_ext = Y_ext[ind[0:int(len(data_raw)*0.6)]]
    Y_val_ext = Y_ext[ind[int(len(data_raw)*0.6):int(len(data_raw)*0.8)]]
    Y_test_ext = Y_ext[ind[int(len(data_raw)*0.8):]]
    
    I = len(X_train)

    # MMFA Train
    K = 10
    model = MMFA(I, Du, Mu, K)
    model.modelparams.alpha = 0.00
    model.modelparams.Xon = 1
    model.fit(X_train.T, Y_train.T, X_val.T, Y_val.T, Y_val_ext, epochno = 10)
    
    latentparams = model.latent_params(model.modelparams, X_test.shape[0])
    model.e_step(model.modelparams, latentparams, X_test.T, Y_test.T, model.MM)
    
    Pred_r = (model.modelparams.W @ latentparams.U_mean).T
    Cov = model.modelparams.Sigma_x
    LL_MMFA = -(len(X_test) * np.log(np.linalg.det(Cov)))/2 - np.sum((X_test - Pred_r).T * (np.linalg.inv(Cov) @ (X_test - Pred_r).T))/2 - (len(X_test) * X_test.shape[1]) /2
    
    prob = np.empty((Y_test_ext.shape[0], 0))
    ind = 0
    for i in model.modelparams.M:
        ind2 = ind + i
        prob_tmp = softmax((model.modelparams.H @ latentparams.U_mean)[ind:ind2, :]).T
        prob_tmp = np.append(prob_tmp[:, -1][None].T, prob_tmp[:, 0:-1], axis = 1)
        prob = np.append(prob, prob_tmp, axis = 1)
        ind = ind + i   
    
    LL_MMFA = LL_MMFA + np.sum(Y_test_ext * np.log(prob))
    Perp_MMFA = np.exp(-LL_MMFA / (len(X_test)))
    print("Test Likelihood of MMFA:", LL_MMFA)
    print("Perplexity of MMFA:", Perp_MMFA)
    LL_MMFA_vec.append(LL_MMFA)
    Perp_MMFA_vec.append(Perp_MMFA)
    
print("Mean test likelihood of MMFA:", np.mean(LL_MMFA_vec))
print("Std test likelihood of MMFA:", np.std(Perp_MMFA))