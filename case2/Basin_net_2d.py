# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 17:59:21 2020

@author: cheyi
"""

import os 

# please modifify the path accordingly
# the path is used to load data
path_file = r'G:\My Drive' 
path100 = r'G:\My Drive'

import numpy as np
import pylab as py
from sklearn_rvm import EMRVC
from scipy.special import expit
import glob
import time
from sklearn.model_selection import GridSearchCV
import networkx as nx
import warnings
warnings.filterwarnings("ignore")

def get_genindex():
    np.random.seed(10)
        
    adj = np.genfromtxt('adj_IEEE_true.csv', delimiter = ',', skip_header = 1)
    # network topology
    G = nx.from_numpy_matrix(adj)
    # random select half gen. and half loads
    N = G.number_of_nodes()
    node_list = list(range(N))
    np.random.shuffle(node_list)
    load_index = sorted(node_list[:N//2])
    gen_index = sorted(node_list[N//2:])
    return np.array(gen_index)

# data
STATES_100 = np.load(os.path.join(path_file,'STATE_100_RKA.npy'))
BASIN_100 = np.load(os.path.join(path_file,'BASIN_ARRAY_100_RKA.npy'))
basin_i_100 = np.mean(BASIN_100, axis = 1)

theta_g_ = np.linspace(-np.pi, np.pi, 100)
theta_gp_ = np.linspace(-10, 10, 100)
theta_g, theta_gp = np.meshgrid(theta_g_, theta_gp_)
theta_g = theta_g.reshape((-1,1))
theta_gp = theta_gp.reshape((-1,1))

STATES_10000 = np.hstack([theta_g, theta_gp])
files = glob.glob(os.path.join(path100, 'BASIN_ARRAY_10000_RKA_*.npy'))
files = sorted(files, key = lambda f:int(f.split('/')[-1].split('_')[-1].split('.')[0]))
BASIN_10000 = np.array([np.load(file) for file in files])
basin_i_10000 = np.mean(BASIN_10000, axis = 1)

#py.plot(basin_i_100, 'r-o')
#py.plot(basin_i_10000, 'g-o')

def K_center_greedy(X_boundary, X, K = 20):
    n = len(X_boundary)
    DIS = np.zeros(n)
    IND = np.zeros(K, dtype = np.int)
    for k in range(K):
        for i in range(n):
            dis = np.min(np.linalg.norm(X - X_boundary[i], axis = 1))
            DIS[i] = dis
        ind = np.argmax(DIS)
        X = np.vstack((X, X_boundary[ind]))
        IND[k] = ind
    return IND

def on_boundary(rvm, X_test, y_test, n_std = 2):
    Phi = rvm._get_kernel(X_test, rvm.relevance_vectors_) / rvm._scale
    if rvm.bias_used:
        Phi = np.hstack([np.ones((Phi.shape[0], 1)), Phi]) 
    f = np.dot(Phi, rvm.mu_)
    X_var = Phi@rvm.Sigma_@Phi.T
    X_std = np.sqrt(np.diag(X_var))
    upperp = expit(f+n_std*X_std)
    lowerp = expit(f-n_std*X_std)
    
    isboundary = [True if i <= 0.5 <= j else False for i,j in zip(lowerp, upperp)]
    X_boundary = X_test[isboundary]
    y_boundary = y_test[isboundary]
    
    return X_boundary, y_boundary

def converge_check(rvm, X_test, b_last, ep = 0.05):
    y_pred = rvm.predict(X_test)
    y_prob = rvm.predict_proba(X_test)
    y_pred_bs = np.mean(y_pred)
    impro = np.abs(b_last - y_pred_bs)
    b_last = y_pred_bs
    if impro <= ep:
        converge = True
    else:
        converge = False
    return converge, b_last, y_pred, y_prob

def main_single(i, ep, n_std, K, plot = False):
    
    X_train = STATES_100[i].copy()
    y_train = BASIN_100[i].copy()
    
    X_test = STATES_10000
    y_test = BASIN_10000[i]
    
    para = {'gamma':[1, 10, 100]}
    
    try:
        GS = GridSearchCV(EMRVC(kernel = 'rbf', bias_used = True), para, cv = 3, n_jobs = 12)
        GS.fit(X_train, y_train)
        rvm = GS.best_estimator_
    except:
        rvm = EMRVC(kernel = 'rbf', gamma = 1, bias_used = True)
        rvm.fit(X_train, y_train)
    print(f'gamma_ini = {rvm.gamma}')
        
    # Sequential Design
    converge = False
    iters = 0
    b_last = 0
    X_select = []
    while not converge:
        iters += 1
        X_boundary, y_boundary = on_boundary(rvm, X_test, y_test, n_std = n_std)
        if len(X_boundary) != 0:
            ind = K_center_greedy(X_boundary, X_train, K = K)
            X_candidate = X_boundary[ind]
            y_candidate = y_boundary[ind]
            X_select.append(X_candidate)
             
            X_train = np.vstack([X_train, X_candidate])
            y_train = np.concatenate((y_train, y_candidate))
            try:
                GS.fit(X_train, y_train)
                rvm = GS.best_estimator_
                print(f'gamma_{iters} = {rvm.gamma}')
            except:
                rvm = EMRVC(kernel = 'rbf', gamma = 1, bias_used = True)
                rvm.fit(X_train, y_train)
                print('gamma_default')
#            rvm.fit(X_train, y_train)
        else:
            break
        converge, b_last, y_pred, y_prob = converge_check(rvm, X_test, b_last, ep = ep)  
        if plot:
            py.figure()
            n = 100
            py.contourf(theta_g_, theta_gp_, BASIN_10000[i].reshape((n, n)), cmap = 'binary')
#            py.plot(X_boundary[:,0], X_boundary[:,1], 'bo')
            py.plot(X_candidate[:,0], X_candidate[:,1], 'go')
            py.contour(theta_g_, theta_gp_, y_prob[:,0].reshape((n, n)), levels = [0.5], colors = 'r', linewidths = 3)
            py.xlim([-3.14,3.14])
            py.ylim([-10.1,10.1])
            py.xlabel(r'$\theta$', fontsize = 20)
            py.ylabel(r'$\omega$', fontsize = 20)
            py.matplotlib.pyplot.subplots_adjust(left = 0.13, bottom = 0.12)
            
        
    print(f'{iters} is used')
    return y_pred, iters


    
if __name__ == '__main__':
    py.rcParams['xtick.labelsize'] = 20
    py.rcParams['ytick.labelsize'] = 20
    py.rcParams['axes.linewidth'] = 2
    
    ep = 0.001
    n_std = 3
    K = 30
    nodelist = (basin_i_100!=1).nonzero()[0]
    

    #%%
    c_time = 0
    nodelist = (basin_i_100!=1).nonzero()[0]
    Y_PRED = np.zeros(len(nodelist))
    ITERS = [] 
    for i in range(len(nodelist)):
        ind = nodelist[i]
        print(f'node {ind}')
        st = time.time()
        y_pred_i, iters = main_single(ind, ep = ep, n_std = n_std, K = K)
        ITERS.append(iters)
        Y_PRED[i] = np.mean(y_pred_i)
        time_used = time.time() - st
        c_time += time_used
        print(f'error = {np.abs(basin_i_10000[ind] - Y_PRED[i]):.2f}')

#        print(f'r_error = {np.abs(basin_i_10000[ind] - Y_PRED[i])/basin_i_10000[ind]:.2f}')
        print(f'time used = {time_used:.2f}s')
        print(f'cumulated time used = {c_time:.2f}s\n')
    
    
    ERROR_pred = np.abs(basin_i_10000[nodelist] - Y_PRED)/basin_i_10000[nodelist]
    ERROR_pred2 = np.abs(basin_i_10000[nodelist] - Y_PRED)
    

    
    py.rcParams['xtick.labelsize'] = 15
    py.rcParams['ytick.labelsize'] = 15
    py.rcParams['axes.linewidth'] = 2
    
    py.figure()
    py.plot(basin_i_10000[nodelist], 'b-o', linewidth = 3, markersize = 10)
    py.plot(Y_PRED, 'r-o', linewidth = 3, markersize = 10)
    py.xticks([])
    py.legend(['True BS', 'Predicted BS'], fontsize = 15)
    py.xlabel('generator index', fontsize = 20)
    py.ylabel('BS', fontsize = 20)
    py.xlim([0,31])
    
#    py.figure()
#    py.plot(ERROR_pred2, 'r-o', linewidth = 2, markersize = 10)
#    py.xticks([])
#    py.xlabel('generator', fontsize = 20)
#    py.ylabel('absolute error', fontsize = 20)
#    py.xlim([0,31])

    
#    py.figure(figsize = (15, 6))
#    py.plot(ERROR_pred, 'bo-')
#    py.xticks(ticks = range(len(nodelist)), labels = nodelist)
#    py.xlabel('generator', fontsize = 20)
#    py.ylabel('e', fontsize = 20)
#    py.xlim([0,31])
    
    py.figure()
    py.plot(ITERS, 'ro-', linewidth = 3, markersize = 10)
    py.yticks([2,6,10,14])
    py.xticks([])
    py.xlabel('generator index', fontsize = 20)
    py.ylabel('iterations', fontsize = 20)
    py.xlim([0,31])
    py.ylim([2,15])
        
  