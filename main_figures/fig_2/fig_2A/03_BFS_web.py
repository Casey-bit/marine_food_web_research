'''
 # Copyright (C) 2022-08-09 School of IoT, Jiangnan University Limited
 #
 # All Rights Reserved.
 #
 # Paper: Successive Relocation of Marine Species in the Northern Hemisphere from 1970 to 2020
 # First author: Zhenkai Wu
 # Corresponding author: Ya Guo, E-mail: guoy@jiangnan.edu.cn
'''
import numpy as np
import pandas as pd
import pickle
from queue import Queue
import networkx as nx
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

font1 = {
        'weight': 'bold',
        'style':'normal',
        'size': 20,
        }
font2 = {
        'weight': 'bold',
        'style':'normal',
        'size': 15,
        }

def set_axis(ax):
    ax.grid(False)
    ax.set_xlim(1961,2022)
    ax.set_ylim([18,65])
    ax.set_xticks([1970,1980,1990,2000,2010,2020])
    ax.set_yticks([20,30,40,50,60])
    ax.set_xticklabels(ax.get_xticks(),font1)
    ax.set_yticklabels(ax.get_yticks(),font1)
    ax.set_xlabel('Year',font1)
    ax.set_ylabel('Latitude (°N)',font1)
    ax.spines['top'].set_linewidth('2.0')
    ax.spines['bottom'].set_linewidth('2.0')
    ax.spines['left'].set_linewidth('2.0')
    ax.spines['right'].set_linewidth('2.0')
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    for tickline in ax.xaxis.get_ticklines():
        tickline.set_visible(True)
    for tickline in ax.yaxis.get_ticklines():
        tickline.set_visible(True)
    ax.tick_params(which="major",
                    length=15,width = 2.0,
                    colors = "black",
                    direction = 'in',
                    tick2On = False)


'''
web
'''
correlation = load_variavle(r'chugao\fish_220114_re\correlation_median_point\correlation.txt')
pvalue = load_variavle(r'chugao\fish_220114_re\correlation_median_point\pvalue.txt')
familyExtract = load_variavle(r'chugao\fish_220114_re\correlation_median_point\familyExtract.txt') # family name
medianPoint = load_variavle(r'chugao\fish_220114_re\correlation_median_point\median_denoising.txt') # median
reserve = load_variavle(r'chugao\fish_220114_re\correlation_median_point\reserve.txt') # new - old
# medianPoint = np.array([[0.0 for year in range(51)] for family in range(2846)])

# latitude = np.array(load_variavle(r'chugao\fish_220114_re\cluster\latitudeByYear_1970_2020_2846_1.txt'))
# print(latitude.shape)

# for family in range(2846):
#     for year in range(51):
#         medianPoint[family, year] = np.mean(latitude[year, family, 0])
#     print(family)


# for family in range(medianPoint.shape[0]):
#     for year in range(medianPoint.shape[1] - 9):
#         medianPoint[family, year] = np.mean(medianPoint[family, year: year + 10][medianPoint[family, year: year + 10] > 0])

wholegroup = []
wholeedgelist = []

for fam in range(medianPoint.shape[0]):
    if fam == 550:
        break
    edgeList = []
    group = [] # a set of R2 > 0.9
    q = Queue()
    init = fam
    q.put(init)
    group.append(init)

    while not q.empty():
        nowFamily = q.get()
        correlation[nowFamily, nowFamily] = 0
        if len(group) < 8:
            for i in range(len(familyExtract)):
                if correlation[nowFamily, i] > 0.9 and correlation[nowFamily, i] == np.max(correlation[nowFamily]):
                    if i not in group:
                        q.put(i)
                        group.append(i)
                    if (i, nowFamily, "%0.2f" % correlation[nowFamily, i]) not in edgeList:
                        edgeList.append((nowFamily, i, "%0.2f" % correlation[nowFamily, i]))
                    # correlation[nowFamily, i] = 0
                    break
            for i in range(len(familyExtract)):    
                if correlation[nowFamily, i] < -0.9 and correlation[nowFamily, i] == np.min(correlation[nowFamily]):
                    if i not in group:
                        q.put(i)
                        group.append(i)
                    if (i, nowFamily, "%0.2f" % correlation[nowFamily, i]) not in edgeList:
                        edgeList.append((nowFamily, i, "%0.2f" % correlation[nowFamily, i]))
                    # correlation[nowFamily, i] = 0
                    break
                
    print(fam)
    for item in group:
        if item not in wholegroup:
            wholegroup.append(item)
    for s,t,w in edgeList:
        if (s,t,w) not in wholeedgelist and (t,s,w) not in wholeedgelist:
            wholeedgelist.append((s,t,w))

'''
web visulization
'''
cluster_inf = load_variavle(r'chugao\fish_220114_re\figure3\cluster.txt')

output = pd.DataFrame({'source':[],'target':[],'weight':[],'label':[]})

group = wholegroup
edgeList = wholeedgelist
G = nx.DiGraph()
G.add_nodes_from(group)

for i,j,k in edgeList:
    G.add_edge(i,j,weight=k)
    if float(k) > 0:
        if cluster_inf[i] == cluster_inf[j]:
            label = cluster_inf[i]
        else:
            label = -1
        output = output.append({'source':i,'target':j,'weight':k,'label':label},ignore_index=True)

for i in range(len(group)):
    for j in range(i + 1, len(group)):
        if correlation[group[i], group[j]] > 0.9 and (group[i],group[j]) not in G.edges and (group[j],group[i]) not in G.edges:
            G.add_edge(group[i],group[j],weight="%0.2f" % correlation[group[i], group[j]])
            if cluster_inf[group[i]] == cluster_inf[group[j]]:
                label = cluster_inf[group[i]]
            else:
                label = -1
            output = output.append({'source':group[i],'target':group[j],'weight':"%0.2f" % correlation[group[i], group[j]],'label':label},ignore_index=True)
            # print(i, j, label)
output.to_csv(r'E:\gephi\data\vertex0.9_cluster.csv')
