import pandas as pd
import scipy
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch 
import scipy.spatial.distance as ssd
from scipy.cluster.vq import vq,kmeans,whiten
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

medianPoint = load_variavle(r'fig2\depth\median_denoising.txt') # median
reserve = load_variavle(r'fig2\depth\reserve.txt') # new - old
correlation = load_variavle(r'fig2\depth\correlation.txt')
# correlation[correlation < 0.9] = 0
distances = 1 - correlation  # pairwise distnces
distArray = np.array([0.0 for i in range(int(571*570/2))])
c = 0
for i in range(571):
    for j in range(i + 1, 571):
        distArray[c] = distances[i, j]
        c += 1


disMat = 100 * distArray
print(disMat)
Z=sch.linkage(disMat,method='average') 
P=sch.dendrogram(Z)
cluster= sch.fcluster(Z, t=90, criterion='distance') 
print("Original cluster by hierarchy clustering:\n",cluster)
l = Counter(cluster).most_common(15)
print(l)

save_variable(cluster,r'fig2\depth\cluster.txt')
np.savetxt(r'E:\gephi\data\label_depth_230127.csv', cluster,fmt='%d', delimiter = ',') 
plt.show()

for i in range(len(cluster)):
    if cluster[i] == 1:
        idx = [o for n, o in reserve if n == i][0]
        plt.plot(medianPoint[idx])
plt.show()
