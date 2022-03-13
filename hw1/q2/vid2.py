#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 19:06:35 2022

@author: divyansh
"""

# # -*- coding: utf-8 -*-
# """
# Spyder Editor

# This is a temporary script file.
# """

import cv2
import numpy as np
import statistics
import random
import matplotlib.pyplot as plt

x_t = []
y_t = []
x_b = []
y_b = []
x = []
y = []
func = []
x_lis = []
pp = []
test = []
ii = 0
# 2488
# 3520 for video 2
cap = cv2.VideoCapture('ball_video2.mp4')
while(cap.isOpened()):
    
    ret, frame = cap.read()
    if ret == False:
        break
    red_channel = frame[:,:,2]
    cons = 255
    red_channel = cons - red_channel

   
    result = np.where((red_channel > 20) )
    # print(np.size(result))
    y_t.append(3519 - result[0][0])
    x_t.append(result[1][0])
    y_b.append(3519 - result[0][-1])
    x_b.append(result[1][-1])

            
        
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break    
    
[x.append((a + b)/2) for a,b in zip(x_t, x_b)]
[y.append((a + b)/2) for a,b in zip(y_t, y_b)]

plt.scatter(x_t, y_t, c = 'b', label = 'Top point')
plt.scatter(x, y, c = 'm', label = 'Center point')

x_sq = np.array(x)**2
x_sq = np.matrix(x_sq)
x_sq_trans = np.transpose(x_sq)
# print(np.shape(x_sq_trans))

x = np.matrix(x)
x_trans = np.transpose(x)
print(np.shape(x_trans))

one = [1] * 24
one = np.matrix(one)
one_trans = np.transpose(one)
# print(np.shape(one_trans))

A = np.hstack((x_sq_trans, x_trans, one_trans))
# print(A)
# print(np.shape(A))
A_t = np.transpose(A)
invv = np.matmul(A_t, A)
A_inv = np.linalg.inv(invv) 
xx = np.matmul(A_inv, A_t)
xxxx = np.matmul(xx, y_t)
print (xxxx)

# x = x.astype(int)
# print(xxxx[0,0])

test = range(3520)
# print(type(test))

test = np.array(test)



for ii in range(3520):
    
    yy = xxxx[0, 0].round(10) * test[ii]* test[ii] + xxxx[0, 1].round(10) * test[ii] + xxxx[0, 2].round(10)
    func.append(yy)

 

plt.scatter(x_b, y_b, c = 'r', label = 'Bottom point')
plt.plot(test, func, c='g', label = 'Fitted curve')
plt.legend(loc='upper left');
plt.title('Noisy')
plt.show()    

cap.release()
cv2.destroyAllWindows()        
