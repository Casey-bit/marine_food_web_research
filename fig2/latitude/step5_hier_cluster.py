import pandas as pd
import scipy
from collections import Counter #引入Counter
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch #用于进行层次聚类，画层次聚类图的工具包
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
    dist = np.sqrt(np.sum(np.square(x-y)))   # 注意：np.array 类型的数据可以直接进行向量、矩阵加减运算。np.square 是对每个元素求平均~~~~
    return dist

medianPoint = load_variavle(r'fig2\latitude\median_denoising.txt') # median
reserve = load_variavle(r'fig2\latitude\reserve.txt') # new - old
correlation = load_variavle(r'fig2\latitude\correlation.txt')
# correlation[correlation < 0.9] = 0
distances = 1 - correlation  # pairwise distnces
distArray = np.array([0.0 for i in range(int(571*570/2))])
c = 0
for i in range(571):
    for j in range(i + 1, 571):
        distArray[c] = distances[i, j]
        c += 1

# distArray = ssd.squareform(distances)  # scipy converts matrix to 1d array
# print(len(distArray), 571*558/2)
#生成待聚类的数据点,这里生成了20个点,每个点4维:
# points=scipy.randn(5,4)  

# #1. 层次聚类
# #生成点与点之间的距离矩阵,这里用的欧氏距离:
# disMat = sch.distance.pdist(points,'euclidean') 
disMat = 100 * distArray
print(disMat)
#进行层次聚类:
Z=sch.linkage(disMat,method='average') 
#将层级聚类结果以树状图表示出来并保存为plot_dendrogram.png
P=sch.dendrogram(Z)
cluster= sch.fcluster(Z, t=70, criterion='distance') 
print("Original cluster by hierarchy clustering:\n",cluster)
l = Counter(cluster).most_common(15)
print(l)

save_variable(cluster,r'fig2\latitude\cluster.txt') # 存每个family所在点类别
np.savetxt(r'E:\gephi\data\label_220125.csv', cluster,fmt='%d', delimiter = ',') 
plt.show()

for i in range(len(cluster)):
    if cluster[i] == 3:
        idx = [o for n, o in reserve if n == i][0]
        plt.plot(medianPoint[idx])
plt.show()

# data = pd.read_csv(r'E:\gephi\data\vertex0.9_node_220125.csv')
# data = data.values.tolist()
# data = np.array(data).T

# color = ['r','g','b','m','orange']

# avet = np.array([[0.0 for year in range(51)] for lev in range(5)])

# for k in range(5):
#     ax = [plt.subplot(2,5,i) for i in range(1,6)]
#     level = np.array([[0.0 for year in range(51)] for l in range(5)])
#     count = np.array([0 for l in range(5)])
#     fam = [[] for l in range(5)] # 5 level old
#     for new in range(571):
#         if cluster[new] == l[k][0]:
#             old = [o for n, o in reserve if n == new][0]
#             fam[data[2,new]-1].append(old)
#             # ax[data[2,new]-1].plot(np.arange(0,51,1), medianPoint[old], color = color[data[2,new]-1])
#             level[data[2,new]-1] += medianPoint[old]
#             count[data[2,new]-1] += 1
#     # for lev in range(5):
#     #     if count[lev] > 0:
#     #         level[lev] /= count[lev]
#     #     plt.plot(np.arange(0,51,1), level[lev], color = color[lev], label = str(lev + 1))
#     # plt.legend()
    
#     for lev in range(5):
#         std = np.array([0.0 for year in range(51)])
#         ave = np.array([0.0 for year in range(51)])
#         for year in range(51):
#             std[year] = np.std([medianPoint[o][year] for o in fam[lev]], ddof=1)
#             ave[year] = np.mean([medianPoint[o][year] for o in fam[lev]])
#         avet[lev] += ave / 5
#         if k < 3:
#             ax[lev].plot(np.arange(1970,1970 + 51,1), ave, color = color[lev])
#             ax[lev].fill_between(np.arange(1970,1970 + 51,1), ave - std / 2, ave+ std / 2,facecolor = color[lev], alpha = 0.25)

#     # plt.suptitle(l[k][0])
#     for i in range(5):
#         ax[i].set_ylim([40,60])
#         ax[i].grid(True)
# ax6 = plt.subplot(2,1,2)
# for lev in range(5):     
#     ax6.plot(np.arange(1970,1970 + 51,1), avet[lev], color = color[lev],label = str(lev + 1))
# ax6.legend()
# plt.show()
# plt.savefig('plot_dendrogram.png')
#根据linkage matrix Z得到聚类结果:


# #2. k-means聚类
# #将原始数据做归一化处理
# data=whiten(points)

# #使用kmeans函数进行聚类,输入第一维为数据,第二维为聚类个数k.
# #有些时候我们可能不知道最终究竟聚成多少类,一个办法是用层次聚类的结果进行初始化.当然也可以直接输入某个数值. 
# #k-means最后输出的结果其实是两维的,第一维是聚类中心,第二维是损失distortion,我们在这里只取第一维,所以最后有个[0]
# centroid=kmeans(data,max(cluster))[0]  

# #使用vq函数根据聚类中心对所有数据进行分类,vq的输出也是两维的,[0]表示的是所有数据的label
# label=vq(data,centroid)[0] 
# print("Final clustering by k-means:\n",label)
