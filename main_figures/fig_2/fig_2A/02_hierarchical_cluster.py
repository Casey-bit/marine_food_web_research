import pandas as pd
from collections import Counter #引入Counter
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch #用于进行层次聚类，画层次聚类图的工具包
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

medianPoint = load_variavle(r'chugao\fish_220114_re\correlation_median_point\median_denoising.txt') # median
reserve = load_variavle(r'chugao\fish_220114_re\correlation_median_point\reserve_family.txt') # new - old
correlation = load_variavle(r'chugao\fish_220114_re\correlation_median_point\correlation.txt')
# correlation[correlation < 0.9] = 0
distances = 1 - correlation  # pairwise distnces
distArray = np.array([0.0 for i in range(int(559*558/2))])
c = 0
for i in range(559):
    for j in range(i + 1, 559):
        distArray[c] = distances[i, j]
        c += 1

# hierarchical clustering
disMat = 100 * distArray
print(disMat)

Z=sch.linkage(disMat,method='average') 

P=sch.dendrogram(Z)
cluster= sch.fcluster(Z, t=70, criterion='distance') 
print("Original cluster by hierarchy clustering:\n",cluster)
l = Counter(cluster).most_common(5)
print(l)

save_variable(cluster,r'chugao\fish_220114_re\figure3\cluster.txt') # the clustering category for each family
np.savetxt(r'E:\gephi\data\label_0523.csv', cluster,fmt='%d', delimiter = ',') 
plt.show()

data = pd.read_csv(r'E:\gephi\data\vertex0.9_node.csv')
data = data.values.tolist()
data = np.array(data).T

color = ['r','g','b','m','orange']

avet = np.array([[0.0 for year in range(51)] for lev in range(5)])

for k in range(5):
    ax = [plt.subplot(2,5,i) for i in range(1,6)]
    level = np.array([[0.0 for year in range(51)] for l in range(5)])
    count = np.array([0 for l in range(5)])
    fam = [[] for l in range(5)] # 5 level old
    for new in range(559):
        if cluster[new] == l[k][0]:
            old = [o for n, o in reserve if n == new][0]
            fam[data[2,new]-1].append(old)

            level[data[2,new]-1] += medianPoint[old]
            count[data[2,new]-1] += 1
    
    for lev in range(5):
        std = np.array([0.0 for year in range(51)])
        ave = np.array([0.0 for year in range(51)])
        for year in range(51):
            std[year] = np.std([medianPoint[o][year] for o in fam[lev]], ddof=1)
            ave[year] = np.mean([medianPoint[o][year] for o in fam[lev]])
        avet[lev] += ave / 5
        if k < 3:
            ax[lev].plot(np.arange(1970,1970 + 51,1), ave, color = color[lev])
            ax[lev].fill_between(np.arange(1970,1970 + 51,1), ave - std / 2, ave+ std / 2,facecolor = color[lev], alpha = 0.25)

    for i in range(5):
        ax[i].set_ylim([40,60])
        ax[i].grid(True)
ax6 = plt.subplot(2,1,2)
for lev in range(5):     
    ax6.plot(np.arange(1970,1970 + 51,1), avet[lev], color = color[lev],label = str(lev + 1))
ax6.legend()
plt.show()