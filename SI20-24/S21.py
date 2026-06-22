

import pandas as pd
import numpy as np
import pickle


df_raw = pd.read_csv('FamilyLat_q2.csv', header=None)
years = list(range(1970, 2021))
df_raw.columns = ['level'] + years

df_raw.insert(0, 'id', range(1, len(df_raw) + 1))
print(df_raw)
data = df_raw.iloc[:, 2:].to_numpy() 
print("shape:", data.shape)

#step3
from statistics import median
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pywt
from tsmoothie.smoother import LowessSmoother
from scipy import stats
import scipy

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





YEAR_START_ALL = 1970
YEAR_END_ALL   = 2020
WINDOW_SIZE    = 40
STEP           = 2
MIN_VALID      = 30 


windows = []
start = YEAR_START_ALL
while start + WINDOW_SIZE - 1 <= YEAR_END_ALL:
    end = start + WINDOW_SIZE - 1
    windows.append((start, end))
    start += STEP

print("allwindow：", windows)

# all_reserve = []  # reserve
# all_loss    = []  #  loss
# all_tot     = []  #  tot
#medianPoint = data

for (START, END) in windows:
    year_idx_start = START - 1970
    year_idx_end   = END   - 1970
    yearTotal = year_idx_end - year_idx_start + 1
    medianPoint = data[:,year_idx_start:year_idx_end+1]
    reserve = []
    loss = []
    tot = []

    for family in range(medianPoint.shape[0]):
        #series = medianPoint[family, year_idx_start:year_idx_end+1]
        series = medianPoint[family,:]

        valid_years = len(series[series > 0])

        ok_10yr = True
        for start_idx in [0,10,20,30]:
            sub = series[start_idx:start_idx+10]
            if len(sub[sub > 0]) == 0:
                ok_10yr = False
                break
        
        if not(ok_10yr and valid_years > MIN_VALID):
            continue
        
        
        num = 0
        for elem in series:
            if not elem > 0:
                num += 1
            elif num:
                tot.append(num)
                num = 0
        if num:
            tot.append(num)

        # loss
        loss.append(len(series) - valid_years)

        reserve.append(family)

    rank_result = list(enumerate(reserve))
    print(rank_result)
    
    reserve = rank_result
    
    family = df_raw[['id','level']]
    print(family)
    
    family.to_csv(r'vertex0.9_node_230730.csv')
    
    corre = []
    def wavelet_denoising(data):
        db4 = pywt.Wavelet('db6')
        coeffs = pywt.wavedec(data, db4, level=10)
        coeffs[len(coeffs)-2] *= 0
        coeffs[len(coeffs)-3] *= 0
        coeffs[len(coeffs)-4] *= 0
        meta = pywt.waverec(coeffs, db4)
        c, p = scipy.stats.pearsonr(data, meta[:len(data)])
        corre.append(c)
        return meta[:len(data)]
    
    median_denoising = np.array([[np.nan for year in range(yearTotal)] for family in range(len(family))])
    
    percent_up = 0
    percent_down = 0
    for family in range(medianPoint.shape[0]):
        df = pd.DataFrame(medianPoint[family])
        if family in reserve:
            percent_up += len(df)-len(df.dropna())
            percent_down += len(df)
        df.fillna(df.interpolate(),inplace=True)
        df.fillna(method='backfill',inplace=True)
        medianPoint[family] = np.copy(np.array(df[0].values.tolist()))
    save_variable(medianPoint, r'median_interpolation.txt')
    
    
    for family in range(medianPoint.shape[0]):
        if family in [y for x, y in rank_result]:
            smoother = LowessSmoother(smooth_fraction=0.1, iterations=1)
            nanNum = len(medianPoint[family][np.isnan(medianPoint[family])])
            if nanNum < medianPoint.shape[1]:
                smoother.smooth(medianPoint[family][medianPoint[family] > 0])
                median_denoising[family, nanNum:] = np.copy(smoother.smooth_data[0])
    
    for family in range(medianPoint.shape[0]):  #medianPoint.shape[0]
        if family in [y for x, y in rank_result]:
            flag = False
            for year in range(0, yearTotal - 4):
                for time in range(4):
                    diff = median_denoising[family, year + time + 1] - median_denoising[family, year]
                    if np.fabs(diff) > 5:
                        nanNum = len(median_denoising[family][np.isnan(median_denoising[family])])
                        if nanNum < medianPoint.shape[1]:
                            median_denoising[family, nanNum:] = np.copy(wavelet_denoising(median_denoising[family][~np.isnan(median_denoising[family])]))
                            flag = True
                            break
                if flag:
                    break
    
    save_variable(median_denoising, r'median_denoising.txt')
    
    
    
    
    ########################################
    
    # step4
    import numpy as np
    import pandas as pd
    import scipy
    
    #yearTotal = 2020 - 1970 + 1
    medianPoint = load_variavle(r'median_denoising.txt')
    medianPointExtract = np.array([[0.0 for year in range(yearTotal)] for family in range(len(reserve))])
    for i, j in reserve:
        medianPointExtract[i] = np.copy(medianPoint[j])
    
    pvalue = np.array([[0 for col in range(len(reserve))] for row in range(len(reserve))], dtype=float)
    correlation = np.array([[0 for col in range(len(reserve))] for row in range(len(reserve))], dtype=float)
    for i in range(len(reserve)):
        for j in range(len(reserve)):
            x = medianPointExtract[i][medianPointExtract[i] > 0]
            y = medianPointExtract[j][medianPointExtract[j] > 0]
            if len(x) - len(y) > 0:
                correlation[i, j], pvalue[i, j] = scipy.stats.pearsonr(x[len(x) - len(y):], y)
            else:
                correlation[i, j], pvalue[i, j] = scipy.stats.pearsonr(x, y[len(y) - len(x):])
    save_variable(correlation, r'correlation.txt')
    save_variable(pvalue, r'pvalue.txt')
    ################################
    
    
    #step5
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

    medianPoint = load_variavle(r'median_denoising.txt') # median
    correlation = load_variavle(r'correlation.txt')
    distances = 1 - correlation  # pairwise distnces

    distArray = np.array([0.0 for i in range(int(correlation.shape[0] * (correlation.shape[0] - 1) / 2))])
    c = 0
    num1 = correlation.shape[0]
    for i in range(num1):
        for j in range(i + 1, num1):
            distArray[c] = distances[i, j]
            c += 1
    disMat = 100 * distArray
    print(disMat)
    Z=sch.linkage(disMat,method='average')
    P=sch.dendrogram(Z)
    cluster= sch.fcluster(Z, t=15, criterion='distance')
    print("Original cluster by hierarchy clustering:\n",cluster)
    l = Counter(cluster).most_common(15)

    save_variable(cluster,r'cluster.txt')
    np.savetxt(r'label_230730.csv', cluster,fmt='%d', delimiter = ',')
    #plt.show()

    for i in range(len(cluster)):
        if cluster[i] == 3:
            idx = [o for n, o in reserve if n == i][0]
            #plt.plot(medianPoint[idx])
    #plt.show()

    ############################
    # step6
    from statistics import median
    from turtle import position
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    #from statsmodels.formula.api import ols
    from collections import Counter
    import pickle
    import seaborn as sns
    import scipy
    import operator
    from queue import Queue
    import networkx as nx
    
    font1 = {
        'weight': 'bold',
        'style': 'normal',
        'size': 20,
    }
    font2 = {
        'weight': 'bold',
        'style': 'normal',
        'size': 15,
    }
    
    
    def set_axis(ax):
        ax.grid(False)
        ax.set_xlim(1961, 2022)
        ax.set_ylim([18, 65])
        ax.set_xticks([1970, 1980, 1990, 2000, 2010, 2020])
        ax.set_yticks([20, 30, 40, 50, 60])
        ax.set_xticklabels(ax.get_xticks(), font1)
        ax.set_yticklabels(ax.get_yticks(), font1)
        ax.set_xlabel('Year', font1)
        ax.set_ylabel('Latitude (°N)', font1)
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
                       length=15, width=2.0,
                       colors="black",
                       direction='in',
                       tick2On=False)
    
    
    '''
    web
    '''
    correlation = load_variavle(r'correlation.txt')
    pvalue = load_variavle(r'pvalue.txt')
    medianPoint = load_variavle(r'median_denoising.txt')  # median
    
    wholegroup = []
    wholeedgelist = []
    
    for fam in range(len(reserve)):
        edgeList = []
        group = []
        q = Queue()
        init = fam
        q.put(init)
        group.append(init)
    
        while not q.empty():
            nowFamily = q.get()
            correlation[nowFamily, nowFamily] = 0
            if len(group) < 8:
                for i in range(len(reserve)):
                    if correlation[nowFamily, i] > 0.9 and correlation[nowFamily, i] == np.max(correlation[nowFamily]):
                        if i not in group:
                            q.put(i)
                            group.append(i)
                        if (i, nowFamily, "%0.2f" % correlation[nowFamily, i]) not in edgeList:
                            edgeList.append((nowFamily, i, "%0.2f" % correlation[nowFamily, i]))
                        # correlation[nowFamily, i] = 0
                        break
                for i in range(len(reserve)):
                    if correlation[nowFamily, i] < -0.9 and correlation[nowFamily, i] == np.min(correlation[nowFamily]):
                        if i not in group:
                            q.put(i)
                            group.append(i)
                        if (i, nowFamily, "%0.2f" % correlation[nowFamily, i]) not in edgeList:
                            edgeList.append((nowFamily, i, "%0.2f" % correlation[nowFamily, i]))
                        break
    
        for item in group:
            if item not in wholegroup:
                wholegroup.append(item)
        for s, t, w in edgeList:
            if (s, t, w) not in wholeedgelist and (t, s, w) not in wholeedgelist:
                wholeedgelist.append((s, t, w))
    
    '''
    web visulization
    '''
    cluster_inf = load_variavle(r'cluster.txt')
    output = pd.DataFrame({'source': [], 'target': [], 'weight': [], 'label': []})
    group = wholegroup
    edgeList = wholeedgelist
    G = nx.DiGraph()
    G.add_nodes_from(group)
    
    for i, j, k in edgeList:
        G.add_edge(i, j, weight=k)
        if float(k) > 0:
            if cluster_inf[i] == cluster_inf[j]:
                label = cluster_inf[i]
            else:
                label = -1
            output = output.append({'source': i, 'target': j, 'weight': k, 'label': label}, ignore_index=True)
    
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            if correlation[group[i], group[j]] > 0.9 and (group[i], group[j]) not in G.edges and (
            group[j], group[i]) not in G.edges:
                G.add_edge(group[i], group[j], weight="%0.2f" % correlation[group[i], group[j]])
                if cluster_inf[group[i]] == cluster_inf[group[j]]:
                    label = cluster_inf[group[i]]
                else:
                    label = -1
                output = output.append(
                    {'source': group[i], 'target': group[j], 'weight': "%0.2f" % correlation[group[i], group[j]],
                     'label': label}, ignore_index=True)
    output.to_csv(r'vertex0.9_cluster_230730.csv')
    
    ######################
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    from matplotlib.patches import Polygon  #
    
    from matplotlib.collections import PatchCollection
    from matplotlib.transforms import TransformedBbox, Bbox
    from matplotlib.image import BboxImage
    from matplotlib.legend_handler import HandlerBase
    from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
    import os
    from mk_test import mk_test
    import pickle
    
    font1 = {
        'weight': 'bold',
        'style': 'normal',
        'size': 15,
    }
    font2 = {
        'weight': 'bold',
        'style': 'normal',
        'size': 12,
    }
    font3 = {
        'weight': 'bold',
        'style': 'normal',
        'size': 20,
    }
    
    
    class ImageHandler(HandlerBase):
        def create_artists(
            self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
        ):
    
            sx, sy = self.image_stretch
            bb = Bbox.from_bounds(xdescent - sx, ydescent -
                                  sy, width + sx, height + sy)
            tbb = TransformedBbox(bb, trans)
            image = BboxImage(tbb)
            image.set_data(self.image_data)
            self.update_prop(image, orig_handle, legend)
            return [image]
    
        def set_image(self, image_path, image_stretch=(0, 0)):
            if os.path.exists(image_path):
                self.image_data = plt.imread(image_path)
            self.image_stretch = image_stretch
    
    
    def set_axis(ax):
        ax.grid(False)
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
        for tickline in ax.yaxis.get_minorticklines():
            tickline.set_visible(True)
        ax.tick_params(which="major",
                       length=15, width=2.0,
                       colors="black",
                       direction='in',
                       tick2On=False,
                       label2On=False)
        ax.tick_params(which="minor",
                       length=5, width=1.0,
                       colors="black",
                       direction='in',
                       tick2On=False,
                       label2On=False)
    
        ax.set_ylim([0, 6])
        ax.set_xticklabels(ax.get_xticklabels(), font1)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels([1, 2, 3, 4, 5], font1)
        ax.set_ylabel("Family Trophic Level", font1)
        ax.set_xlabel("Family Shift Category", font1)
    
    
    df = pd.read_csv(r'vertex0.9_cluster_230730.csv')
    node = []
    df['source'] = df['source'].astype(int)
    df['target'] = df['target'].astype(int)
    
    for elem in df['source']:
        if not elem in node:
            node.append(elem)
    
    for elem in df['target']:
        if not elem in node:
            node.append(elem)
    
    median = load_variavle(r'median_denoising.txt')
    
    mk_test_res = pd.DataFrame({'family':[], 'slope':[], 'za':[], 'describe':[], 'mode':[]})
    
    for n, o in reserve:
        if n in node:
            slope, za = mk_test(median[o])
            if za > 1.96:
                if slope > 0:
                    mk_test_res.loc[len(mk_test_res)] = [n, float(slope), float(za), 'Northward', 1]
                if slope < 0:
                    mk_test_res.loc[len(mk_test_res)] = [n, float(slope), float(za), 'Southward', 2]
            else:
                mk_test_res.loc[len(mk_test_res)] = [n, float(slope), float(za), 'Mixed', 3]
    
    for i in range(1, 4):
        print(len(mk_test_res[mk_test_res['mode'] == i]))
    
    
    node_df = pd.read_csv(r'vertex0.9_node_230730.csv')
    node_df = node_df[['id', 'level']]
    
    mk_test_res = pd.merge(mk_test_res, node_df, left_on=['family'], right_on=['id'])
    mk_test_res_2 = mk_test_res.copy()
    mk_test_res_2['describe'] = 'Total'
    mk_test_res_2['mode'] = 0
    
    mk_test_res = mk_test_res.append(mk_test_res_2, ignore_index = True)
    
    ax = plt.subplot()
    df = mk_test_res
    print(df)
    total = [df[df['mode'] == 0], df[df['mode'] == 1],
             df[df['mode'] == 2], df[df['mode'] == 3]]
    print(total)
    for idx in range(4):
        level = total[idx]['level'].values.tolist()
        print(level)
        # num = total[idx]['family_num'].values.tolist()
        level_prop = [len([i for i in level if i == 1]) / len(level),
                      len([i for i in level if i == 2]) / len(level),
                      len([i for i in level if i == 3]) / len(level),
                      len([i for i in level if i == 4]) / len(level),
                      len([i for i in level if i == 5]) / len(level)]
        level_prop = [i * 1.5 for i in level_prop]
        data_left = [idx + 1 - i/2 for i in level_prop]
        data_right = [idx + 1 + i/2 for i in level_prop]
        color_list = [(140/255, 185/255, 0/255), (20/255, 178/255, 255/255), (255/255, 194/255, 102/255),
                      (255/255, 112/255, 69/255), (255/255, 26/255, 128/255)]  
    
        axes = []
        for l in range(5):
            axes.append(ax.barh(y=l + 1, width=level_prop[l], left=data_left[l],
                        color=color_list[l], height=0.9, label='{}'.format(l + 1)))  
        if idx == 0:
            s = [plt.scatter(-1, -1) for i in range(5)]
            custom_handler1 = ImageHandler()
            custom_handler1.set_image(r"one.png",image_stretch=(2,2))
            custom_handler2 = ImageHandler()
            custom_handler2.set_image(r"two.png",image_stretch=(4,4))
            custom_handler3 = ImageHandler()
            custom_handler3.set_image(r"three.png",image_stretch=(6,6))
            custom_handler4 = ImageHandler()
            custom_handler4.set_image(r"four.png",image_stretch=(8,8))
            custom_handler5 = ImageHandler()
            custom_handler5.set_image(r"five.png",image_stretch=(10,10))
            handles, labels = ax.get_legend_handles_labels()
            hand = zip(s, axes)
            lg = plt.legend(
                hand,
                labels[:5],
                handler_map={tuple: HandlerTuple(
                    ndivide=None), s[0]: custom_handler1, s[1]: custom_handler2, s[2]: custom_handler3, s[3]: custom_handler4, s[4]: custom_handler5},
                prop = font1,
                bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                mode="expand", borderaxespad=0.,ncol = 5
            )
    
            # leng = plt.legend(prop = font2,ncol = 5,loc = 1, columnspacing = 0.5)
            lg.set_title(title='Trophic level', prop = font1)
    
        polygons = []
        for i in range(5):
            ax.text(
                idx + 1,
                i + 1,
                "%.2f" % (level_prop[i] / 1.5 * 100) + '%',  
                color='black', alpha=0.8, ha="center", va='center',
                fontdict=font2)
    
            if i < 4:
                polygons.append(Polygon(xy=np.array([(data_left[i+1], i + 2 - 0.45),   
                                                    (data_right[i+1],
                                                     i + 2 - 0.45),
                                                    (data_right[i], i + 1 + 0.45),
                                                    (data_left[i], i + 1 + 0.45)])))
        ax.add_collection(PatchCollection(polygons,
                                          facecolor='#e2b0b0',
                                          alpha=0.8))
    #set_axis(ax)
    ax.set_xlim([0.5, 4.5])
    ax.set_ylim([0.3, 5.7])
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(['Total', 'Poleward', 'Equatorward', 'Mixed'], rotation=0)
    ax.set_yticks([1, 2, 3, 4, 5])
    #ax.set_yticklabels([1, 2, 3, 4, 5], font1)
    ax.set_yticklabels([1, 2, 3, 4, 5])
    #ax.text(0.6, 0.3 + 5.4 * 0.95, 'B', font3)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params(axis='x', labelsize=14)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xlabel('Family Shift Category', fontsize=14, fontweight='bold')
    ax.set_ylabel('Family Trophic Level', fontsize=14, fontweight='bold')
    fig = plt.gcf()
    fig.set_size_inches(8, 8)
    # fig.savefig(r'fig2B_lati_230730.pdf', dpi=500)
    # fig.savefig('total', dpi=800)
    
    # 自动命名
    filename = f"fig2B_lati_{START}-{END}.png"
    fig.savefig(filename, dpi=800)
    plt.close(fig) 
    
    plt.show()
    
    import os

    os.remove('correlation.txt')
    # os.remove('countArray.txt')
    os.remove('cluster.txt')
    # os.remove('familyExtract.txt')
    # os.remove('FamilyNum_1.csv')
    os.remove('label_230730.csv')
    os.remove('median_denoising.txt')
    os.remove('median_interpolation.txt')
    # os.remove('medianPoint.txt')
    os.remove('pvalue.txt')
    # os.remove('reserve.txt')
    os.remove('vertex0.9_cluster_230730.csv')
    os.remove('vertex0.9_node_230730.csv')
    # os.remove('family_year_median_df.csv')

