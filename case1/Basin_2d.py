# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 21:42:10 2021

@author: cheyi
"""

import numpy as np
import scipy.integrate
import pylab as py
from sklearn_rvm import EMRVC
from pyDOE import lhs
from scipy.special import expit
import time

def generator_rhs(y, t):
    theta_g, theta_gp, theta_c = y
    
    K = 1.15
    alpha = 0.1
    P = 1
    
    # node 1
    dg = theta_gp
    dgp = -alpha*dg + P + K*np.sin(theta_c - theta_g)
    # node 2
    dc = -P + K*np.sin(theta_g - theta_c)
    return [dg, dgp, dc]

def real_model_ij(TW): 
    ep = 0.1
    t = np.linspace(0, 500, 2000) # time
    y = scipy.integrate.odeint(generator_rhs, TW, t)
    y_last = y[-1]
    dg, dgp, dc = generator_rhs(y_last, t)
    
    if  np.abs(dg) <= ep and np.abs(dc) <= ep:
        basin = 1
    else:
        basin = 0
                
    return basin

def K_center_greedy(X_boundary, X, K = 10):
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

def on_boundary(rvm, X_test, y_test, n_std = 1):
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
    X_boundary_std = X_std[isboundary] 
    return X_boundary, y_boundary, X_boundary_std

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
   
if __name__ == '__main__':
    
    py.rcParams['xtick.labelsize'] = 15
    py.rcParams['ytick.labelsize'] = 15
    py.rcParams['axes.linewidth'] = 2

    
    # simulation
    np.random.seed(10)
    
    n = 200
    
    theta_g = np.linspace(-np.pi, np.pi, n)
    theta_gp = np.linspace(-10, 10, n)
    

    X_test = np.load('X2d.npy')
    y_test = np.load('y2d_115.npy')
    
    py.figure()
    py.contourf(theta_g, theta_gp, y_test.reshape((n, n)), cmap = 'binary')
    #%% Space Filling
    
    start_time = time.time()
    LB = np.array([-np.pi, -10])
    UB = np.array([np.pi, 10])
    design_rv_range = np.c_[LB, UB].T
    
    X = lhs(2, samples = 100, criterion = 'maximin')
    X = (design_rv_range[-1] - design_rv_range[0]) * X + design_rv_range[0]
    # multiprocess can be used here
    y = np.array([real_model_ij((X[i][0], X[i][1], 0)) for i in range(X.shape[0])])
    
    X_ = X.copy()
    y_ = y.copy()
    #%% RVM Initialization

    rvm = EMRVC(kernel = 'rbf', gamma = 1, bias_used = True)
    rvm.fit(X_, y_)
    #%% Active Learning
    CANDIDATES = []
    X_BOUNDARY = []
    Y_PRED = []
    Y_PROB = []
    STD = []
    converge = False
    iters = 0
    b_last = 0
    ep = 0.0001
    while not converge:
        n_std = 3
        iters += 1
        print(f'At iteration {iters} \n')
        Si = X_test
        CANi = []
        
        X_boundary, y_boundary, std = on_boundary(rvm, X_test, y_test, n_std = n_std)
        X_BOUNDARY.append(X_boundary)
        ind = K_center_greedy(X_boundary, X_, K = 30)
        
        X_candidate = X_boundary[ind]
        y_canidata = y_boundary[ind]
        
        X_ = np.vstack([X_, X_candidate])
        y_ = np.concatenate((y_, y_canidata))
        
        rvm.fit(X_, y_)
            
        converge, b_last, y_pred, y_prob = converge_check(rvm, X_test, b_last, ep = ep)  

        Y_PRED.append(y_pred)
        Y_PROB.append(y_prob)
        CANDIDATES.append(X_candidate)
        STD.append(std)

    duration = time.time() - start_time 
    print(f'duration = {duration/60}')
     
   #%% plot
    for _ in range(iters):
        py.figure()
        py.contourf(theta_g, theta_gp, y_test.reshape((n, n)), cmap = 'binary')
        py.contour(theta_g, theta_gp, Y_PROB[_][:,0].reshape((n, n)), levels = [0.5], colors = 'r', linewidths = 3)
        py.plot(CANDIDATES[_][:,0], CANDIDATES[_][:,1], 'go')
        py.xlim([-3.14,3.14])
        py.ylim([-10.1,10.1])
        py.xlabel(r'$\theta$', fontsize = 20)
        py.ylabel(r'$\omega$', fontsize = 20)
        py.matplotlib.pyplot.subplots_adjust(left = 0.13, bottom = 0.12)
