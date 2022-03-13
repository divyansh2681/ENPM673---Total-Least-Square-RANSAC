#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 11:26:30 2022

@author: divyansh
"""

import numpy as np  
from pandas import read_csv
import matplotlib.pyplot as plt
import random
import math
# age = []
# cost = []
# sum_age = 0
# sum_cost = 0
# cov_00 = 0
# cov_01 = 0
# cov_10 = 0
# cov_11 = 0

# i = 0
# ii = 0
# out = []
df = read_csv('/home/divyansh/Downloads/ENPM673_hw1_linear_regression_dataset - Sheet1.csv')
# df = read_csv('ENPM673_hw1_linear_regression_dataset - Sheet1.csv')
data = df.values
def my_function(data):
    age = []
    cost = []
    sum_age = 0
    sum_cost = 0
    cov_00 = 0
    cov_01 = 0
    cov_10 = 0
    cov_11 = 0

    i = 0
    ii = 0
    out = []
    
    data_size = len(data)
    # print(data_size)
    
    for i in range(data_size):
        age.append(data[i, 0])
        sum_age = sum_age + age[i]
        cost.append(data[i, 6])
        sum_cost = sum_cost + cost[i]
        i = i + 1
         
    plt.scatter(age, cost, c = 'g', label = 'raw data')    
    avg_age = (sum_age / data_size)    
    # print(avg_age)
    
    avg_cost = (sum_cost / data_size)    
    # print(avg_cost)
    for i in range(data_size):
        cov_00 = cov_00 + (age[i] - avg_age) * (age[i] - avg_age)
        cov_11 = cov_11 + (cost[i] - avg_cost) * (cost[i] - avg_cost)
        cov_10 = cov_10 + (cost[i] - avg_cost) * (age[i] - avg_age)
        cov_01 = cov_01 + (age[i] - avg_age) * (cost[i] - avg_cost)
        i = i + 1
        
    cov_00 = (cov_00 / data_size)
      
    cov_11 = (cov_11 / data_size)    
    
    cov_01 = (cov_01 / data_size) 
        
    cov_10 = (cov_10 / data_size) 
        
    covv = [[cov_00, cov_01], [cov_10, cov_11]]
    # covv = np.matrix([[cov_00, cov_01], [cov_10, cov_11]])
    print(covv)
    
    eigen_values, eigen_vectors = np.linalg.eig(covv)
    
    origin = [np.mean(age), np.mean(cost)]
    
    eig_vec1 = eigen_vectors[:,0]
    eig_vec2 = eigen_vectors[:,1]
    
    print(eigen_values)
    ##################################################
    
    # least square method
    
    one = [1] * data_size
    one = np.matrix(one)
    one_trans = np.transpose(one)
    
    age = np.matrix(age)
    age_trans = np.transpose(age)
    
    cost = np.matrix(cost)
    cost_trans = np.transpose(cost)
    
    A = np.hstack((age_trans, one_trans))
    # print(A)
    A_t = np.transpose(A)
    A_t_times_A = np.matmul(A_t, A)
    A_inv = np.linalg.inv(A_t_times_A)
    
    A_inv_times_At = np.matmul(A_inv, A_t)
    final = np.matmul(A_inv_times_At, cost_trans)
    
    test = range(20, 70)
    test = np.array(test)
    
    for ii in range(50):
        
        yy = final[0, 0].round(10) * test[ii] + final[1, 0].round(10)
        out.append(yy)
      
    # plt.quiver(*origin, *(eig_vec1 * eigen_values[0]), color=['r'], scale=21, label = 'eig 1')
    # plt.quiver(*origin, *(eig_vec2 * eigen_values[1]), color=['b'], scale=21, label = 'eig 2')
    
    plt.quiver(*origin, *(eig_vec1), color=['r'], scale=21, label = 'eig 1')
    plt.quiver(*origin, *(eig_vec2), color=['b'], scale=21, label = 'eig 2')
    # plt.scatter(test, out, c='m', label = 'Linear Least Square')    
    
    
    
    #############################################################
    
    # total least square method
    
    age_err = []
    for i in range(data_size):
        age_err.append(avg_age - age[0, i])
    age_err = np.matrix(age_err)    
    
    cost_err = []
    for i in range(data_size):
        cost_err.append(avg_cost - cost[0, i])
    cost_err = np.matrix(cost_err)    
    
    A_err = np.hstack((age_err.transpose(), cost_err.transpose()))
    
    B = np.matmul(np.transpose(A_err), A_err)
    B_t = np.transpose(B)
    P = np.matmul(B, B_t)
    Q = np.matmul(B_t, B)
    eig_val_P, eig_vec_P = np.linalg.eig(P)
    
    eig_val_Q, eig_vec_Q = np.linalg.eig(Q)
    
    index = np.argmin(eig_val_Q)
    a = eig_vec_Q[0, index]
    b = eig_vec_Q[1, index]
    c = (a * avg_age) + (b * avg_cost)
    
    
    out_err = []
    
    for ii in range(50):
        
        yy = (c - (a * test[ii]) ) / b
        out_err.append(yy)
    out_err = np.array(out_err)
    
    # plt.scatter(test, out_err, c='c', label = 'Total Least Square')  
    plt.legend(loc='upper left');
    plt.show()
    return print(eig_vec_Q[:,0], c)
my_function(data)


########################################################


