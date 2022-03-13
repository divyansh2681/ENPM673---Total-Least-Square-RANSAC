#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 15:24:03 2022

@author: divyansh
"""

import numpy as np  
from pandas import read_csv
import matplotlib.pyplot as plt
import random
import math

# df = read_csv('ENPM673_hw1_linear_regression_dataset - Sheet1.csv')
df = read_csv('/home/divyansh/Downloads/ENPM673_hw1_linear_regression_dataset - Sheet1.csv')
data = df.values

age = []
cost = []
sum_age = 0
sum_cost = 0
iterations = 0
i = 0
ii = 0
out = []
outlier_thrs = 30

data_size = len(data)
# print(data_size)

for i in range(data_size):
    age.append(data[i, 0])
    sum_age = sum_age + age[i]
    cost.append(data[i, 6])
    sum_cost = sum_cost + cost[i]
    i = i + 1
    
age_plot = age  
cost_plot = cost  
age = np.matrix(age)
age_trans = np.transpose(age)

cost = np.matrix(cost)
cost_trans = np.transpose(cost)

A = np.hstack((age_trans, cost_trans))
# print(np.shape(A))
while (iterations<math.inf):
    point = 0
    random1 = np.random.randint(0, 325)
    random2 = np.random.randint(0, 325)
    
    point1 = A[random1]
    point2 = A[random2]
    newdata = [[point1[0, 0], point1[0, 1]], [point2[0, 0], point2[0, 1]]]
    
    newdata = np.matrix(newdata)
    
    # print(type(newdata))
    age_avg = (point1[0, 0] + point2[0, 0]) / 2
    # print(type(age_avg))
    
    cost_avg = (point1[0, 1] + point2[0, 1]) / 2
    # print(cost_avg)
    
    age_err = []
    for i in range(2):
        age_err.append(age_avg - newdata[i, 0])
    age_err = np.matrix(age_err)    
    
    cost_err = []
    for i in range(2):
        cost_err.append(cost_avg - newdata[i, 1])
    cost_err = np.matrix(cost_err)    
    
    A_err = np.hstack((age_err.transpose(), cost_err.transpose()))
    print(np.shape(A_err))
    
    B = np.matmul(np.transpose(A_err), A_err)
    B_t = np.transpose(B)
    P = np.matmul(B, B_t)
    Q = np.matmul(B_t, B)
    eig_val_P, eig_vec_P = np.linalg.eig(P)
    
    eig_val_Q, eig_vec_Q = np.linalg.eig(Q)
    index = np.argmin(eig_val_Q)
    a = eig_vec_Q[0, index]
    b = eig_vec_Q[1, index]
    c = (a * age_avg) + (b * cost_avg)
    test = range(20, 70)
    test = np.array(test)
    
    out_err = []
    
    for ii in range(50):
        
        yy = (c - (a * test[ii]) ) / b
        out_err.append(yy)
    out_err = np.array(out_err)
    iterations = iterations + 1
   
    
    error = np.square(a*age_trans + b*cost_trans - c)
    # print(np.shape(error))
    for i in range(325):
        if (error[i] > outlier_thrs):
            point = point + 0
        else:
            point = point + 1
            
    print(point)   
    accuracy = point / 325 
    
    if (accuracy > 0.6):
        break
print(iterations)

plt.scatter(age_plot, cost_plot, c = 'g', label = 'raw data')           
plt.scatter(test, out_err, c='c', label = 'Ransac')
plt.legend(loc='upper left')
plt.show()
# print(type(cost_plot))
        
