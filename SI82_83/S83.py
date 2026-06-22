# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 14:48:07 2024

@author: GuoWei
"""

from collections import Counter
import matplotlib.pyplot as plt

import scipy.cluster.hierarchy as sch 

import numpy as np
import pickle


def save_variable(v,filename):
    f=open(filename,'wb')
    pickle.dump(v,f)
    f.close()
    return filename

def load_variavle(filename):
    f=open(filename,'rb')
    r=pickle.load(f)
    f.close()
    return r
def calEuclidean(x, y):
    dist = np.sqrt(np.sum(np.square(x-y)))  
    return dist
medianPoint = load_variavle(r'median_denoising.txt') # median
reserve = load_variavle(r'reserve.txt') # new - old
correlation = load_variavle(r'correlation.txt')
distances = 1 - correlation  # pairwise distnces

distArray = np.array([0.0 for i in range(int(correlation.shape[0] * (correlation.shape[0] - 1) / 2))])
c = 0
num1 = correlation.shape[0]
for i in range(num1):  
    for j in range(i + 1, num1):
        distArray[c] = distances[i, j]
        c += 1
        
ax = plt.gca()  # 获取当前轴对象        
        
        
disMat = 100 * distArray
print(disMat)
Z=sch.linkage(disMat,method='average') 
P=sch.dendrogram(Z)
cluster= sch.fcluster(Z, t=15, criterion='distance') 
print("Original cluster by hierarchy clustering:\n",cluster)
l = Counter(cluster).most_common(15)

ax.tick_params(axis='both', labelsize=12)

ax.spines['top'].set_linewidth(2)    # 设置顶部坐标轴线宽度
ax.spines['bottom'].set_linewidth(2) # 设置底部坐标轴线宽度
ax.spines['left'].set_linewidth(2)   # 设置左边坐标轴线宽度
ax.spines['right'].set_linewidth(2)  # 设置右边坐标轴线宽度

#save_variable(cluster,r'cluster.txt') # 存每个family所在点类别
#np.savetxt(r'label_230730.csv', cluster,fmt='%d', delimiter = ',')
plt.show()