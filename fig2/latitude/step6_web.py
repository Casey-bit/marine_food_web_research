from statistics import median
from turtle import position
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from collections import Counter 
import pickle
import seaborn as sns
import scipy
import operator
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
correlation = load_variavle(r'fig2\latitude\correlation.txt')
pvalue = load_variavle(r'fig2\latitude\pvalue.txt')
familyExtract = load_variavle(r'fig2\latitude\familyExtract.txt') # family name
medianPoint = load_variavle(r'fig2\latitude\median_denoising.txt') # median
reserve = load_variavle(r'fig2\latitude\reserve.txt') # new - old
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
    if fam == 570:
        break
    edgeList = []
    # for fam in range(len(familyExtract)):
    group = [] # R2 > 0.9
    q = Queue()
    # init = [new for new, old in reserve if old == 1018][0]
    init = fam
    q.put(init)
    group.append(init)
    # while not q.empty():
    #     nowFamily = q.get()
    #     for i in range(len(familyExtract)):
    #         if (i not in group) and correlation[nowFamily, i] > 0.9:
    #             q.put(i)
    #             group.append(i)
    #             edgeList.append((nowFamily, i))
    #         elif correlation[nowFamily, i] > 0.982:
    #             edgeList.append((nowFamily, i))

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
                
    # print([familyExtract[i] for i in group])
    # # print(np.sort(group))
    # print(len(group))
    print(fam)
    for item in group:
        if item not in wholegroup:
            wholegroup.append(item)
    for s,t,w in edgeList:
        if (s,t,w) not in wholeedgelist and (t,s,w) not in wholeedgelist:
            wholeedgelist.append((s,t,w))
    # fig,ax1 = plt.subplots()
    # color = ['b','g','r','c','m','y','orange','brown','gray','k']
    # if medianPoint[[old for new,old in reserve if new == group[0]][0]][0] < 40:
    #     signalOrder = [] 
    #     for i in range(len(group)):
    #         ax1.plot(np.arange(1970,2021,1),medianPoint[[old for new,old in reserve if new == group[i]][0]][:],label = familyExtract[group[i]][0].split('-')[-1],color=color[i])
    #         signalOrder.append(medianPoint[[old for new,old in reserve if new == group[i]][0]][0])
    #     isBusy = np.zeros(90, dtype=bool) 
    #     for i in range(len(group)):
    #         locationUp = int(signalOrder[i])
    #         locationDown = int(signalOrder[i])
    #         while isBusy[locationUp] and isBusy[locationDown]:
    #             locationUp += 1
    #             locationDown -= 1
    #         if isBusy[locationUp]:
    #             location = locationDown
    #         else:
    #             location = locationUp
    #         ax1.text(1961.5,location, familyExtract[group[i]][0].split('<')[0] + familyExtract[group[i]][0].split('-')[-1],color=color[i])
    #         isBusy[location] = True
        # ax1.legend(ncol = 4, loc = 4, fontsize = 10)
        # set_axis(ax1)
        # # ax1.grid() 
        # ax1.text(1962,64,'K',verticalalignment="top",horizontalalignment="left",fontdict=font1)
        # ax1.text(1965,64,'Generated Food Chain Examples',verticalalignment="top",horizontalalignment="left",fontdict=font1)
        # left,bottom,width,height = [0.2,0.2,0.3,0.2]
        # ax2 = fig.add_axes([left,bottom,width,height])
        # ax2.set_ylim([40,60])
        # for i in range(len(group)):
        #     if group[i] in [265,465]:
        #         ax2.plot(np.arange(1970,2021,1),medianPoint[[old for new,old in reserve if new == group[i]][0]][:],label = familyExtract[group[i]][0].split('-')[-1],color=color[i])
        # # ax2.set_title('Symbiotic relationship',font2)
        # left,bottom,width,height = [0.55,0.2,0.3,0.2]
        # ax3 = fig.add_axes([left,bottom,width,height])
        # ax3.set_ylim([40,60])
        # for i in range(len(group)):
        #     if group[i] in [269,275]:
        #         ax3.plot(np.arange(1970,2021,1),medianPoint[[old for new,old in reserve if new == group[i]][0]][:],label = familyExtract[group[i]][0].split('-')[-1],color=color[i])
        # # ax3.set_title('Predation relationship',font2)
        # ax2.spines['top'].set_color('black')
        # ax2.spines['bottom'].set_color('black')
        # ax2.spines['left'].set_color('black')
        # ax2.spines['right'].set_color('black')
        # ax2.set_xticks([1975,1990,2005,2020])
        # ax2.set_yticks([40,50,60])
        # ax2.set_xticklabels(ax2.get_xticks(),font2)
        # ax2.set_yticklabels(ax2.get_yticks(),font2)
        # ax2.text(2020,41,'Symbiotic Relationship',verticalalignment="bottom",horizontalalignment="right",fontdict=font2)

        # ax3.spines['top'].set_color('black')
        # ax3.spines['bottom'].set_color('black')
        # ax3.spines['left'].set_color('black')
        # ax3.spines['right'].set_color('black')
        # ax3.set_xticks([1975,1990,2005,2020])
        # ax3.set_yticks([40,50,60])
        # ax3.set_xticklabels(ax3.get_xticks(),font2)
        # ax3.set_yticklabels(ax3.get_yticks(),font2)
        # ax3.text(2020,41,'Predation Relationship',verticalalignment="bottom",horizontalalignment="right",fontdict=font2)

        # fig = plt.gcf()
        # fig.set_size_inches(12, 6)
        # fig.savefig(r'D:\VSCode\Code\chugao\fish_220114_re\figure3\K.png',dpi=1000)
        # plt.show()


'''
web visulization
'''
cluster_inf = load_variavle(r'fig2\latitude\cluster.txt')

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
output.to_csv(r'E:\gephi\data\vertex0.9_cluster_230125.csv')
# G = nx.DiGraph(edgeList)

# position = nx.circular_layout(G)

# df = pd.DataFrame(index=G.nodes(), columns=G.nodes())
# for row, data in nx.shortest_path_length(G):
#     for col, dist in data.items():
#         df.loc[row,col] = dist

# df = df.fillna(df.max().max())

# position = nx.kamada_kawai_layout(G, dist=df.to_dict())
# nx.draw_networkx_nodes(G,position,nodelist=group,node_color='r')
# nx.draw_networkx_edges(G,position,arrows=False, edge_color=['r' if (i,j) in [(a,b) for a,b,c in edgeList] else 'k' for i, j in G.edges])
# nx.draw_networkx_labels(G,position)
# weights = nx.get_edge_attributes(G, "weight")
# nx.draw_networkx_edge_labels(G, position, edge_labels=weights,label_pos=0.3)

# # fig = plt.gcf()
# # fig.set_size_inches(72, 40)
# # fig.savefig('chugao\\fish_correlation\\fishnet_R2_{}\\fishnet_{}_{}_{}_R2_{}.png'.format(R2,start,endend,m,R2),dpi=300)
# plt.show()
