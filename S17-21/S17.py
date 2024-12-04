

#step1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter 
import pickle
from functools import reduce
import seaborn as snsa
import scipy
import operator


year11=list(range(1970,1982)) 
year22=list(range(2009,2021))


Time_year=35
window_year =40

data_end=[]
for epoch in range(12): 
    year1=year11[epoch]
    year2=year22[epoch]

    '''
    1. extract Northern Hemisphere
    '''
    total_occurrence = pd.read_csv('data_Occurrence3.csv')
    northern = total_occurrence[total_occurrence['decimallatitude'] > 0]
    '''
    
    '''
    northern['year'] = northern['eventdate'].astype(str).str.slice(0,4)
    northern.drop(columns=['eventdate'], inplace=True)
    northern = northern[northern['year'].astype(int) >= year1]
    northern = northern[northern['year'].astype(int) <= year2]

    northern = northern[northern['bathymetry'] >= 0].reset_index().drop(columns=['index'])
    '''
    
    '''
    northern['count_by_family'] = northern.groupby(['family'])['decimallatitude'].transform('count')
    northern.drop(columns=['Unnamed: 0', 'basisofrecord'], inplace=True)
    northern.dropna(subset=['count_by_family'], inplace=True)
    northern = northern.sort_values(by=['count_by_family'], ascending=False).reset_index().drop(columns=['index'])
    '''
    Remove species with annual records of less than 100
    '''
    northern = northern[northern['count_by_family'].astype(int) >= 100].reset_index().drop(columns=['index'])

    # Calculate the coefficients for each region
    northern['range'] = pd.cut(northern['decimallatitude'], [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90], labels=np.arange(1,19,1))
    year_range_count = northern.groupby(['year','range'])['decimallatitude'].count().reset_index()
    year_range_count['range_mean'] = year_range_count.groupby(['range'])['decimallatitude'].transform('mean')
    year_range_count['corr'] = year_range_count['range_mean'] / year_range_count['decimallatitude']
    year_range_count['corr'][year_range_count['corr'] > 1e8] = 0
    year_range_count = year_range_count[['year','range','corr']]
    
    northern['year'] = northern['year'].astype(int)
    for i in range(3):
        northern['range' + str(i + 1)] = pd.cut(northern['decimallatitude'], [30 * i, 30 + 30 * i], labels=['({},{}]'.format(30 * i, 30 * i + 30)])
        c = northern.groupby(['family'])['range' + str(i + 1)].count().reset_index().rename(columns={'range' + str(i + 1): 'range{}_count'.format(i + 1)})
        northern = pd.merge(northern, c, on=['family'])
        # northern['range{}_count'.format(i + 1)] = northern.groupby(['family'])['range' + str(i + 1)].transform('count')
        northern.drop(columns=['range' + str(i + 1)], inplace=True)
        northern.fillna(value={'range{}_count'.format(i + 1): 0}, inplace=True)

    family_belonging_df = northern.groupby(['family','count_by_family','range1_count','range2_count','range3_count'])['decimallatitude'].count().reset_index().drop(columns=['decimallatitude'])


    family_belonging_df['belonging'] = family_belonging_df[['range1_count','range2_count','range3_count']].idxmax(axis=1)
    family_belonging_df['belonging'] = family_belonging_df['belonging'].astype(str).str.slice(5,6).astype(int)
    print(family_belonging_df)

    
    family_year_median_df = northern.groupby(['family', 'year', 'range'])['decimallatitude'].median().reset_index().rename(columns={"decimallatitude": "median"})
    family_year_median_df.dropna(inplace=True)
    year_range_count['year'] = year_range_count['year'].apply(pd.to_numeric)
    family_year_median_df = family_year_median_df.merge(year_range_count, on=['year','range'])

    family_year_median_df['median_up'] = family_year_median_df['median'] * family_year_median_df['corr']

    family_year_corrsum = family_year_median_df.groupby(['family','year'])['corr'].sum().reset_index()
    family_year_medupsum = family_year_median_df.groupby(['family','year'])['median_up'].sum().reset_index()

    family_year_median_df = pd.merge(family_year_corrsum, family_year_medupsum, on=['family','year'])
    family_year_median_df['median'] = family_year_median_df['median_up'] / family_year_median_df['corr']
    family_year_median_df = family_year_median_df[['family','year','median']]



    family_belonging_df = family_belonging_df[['family', 'belonging','count_by_family']]
    family_year_median_df = pd.merge(family_year_median_df, family_belonging_df, on=['family'])

    family_level = pd.read_csv(r'function_group_2.csv',usecols=(8,13))
    family_level.drop_duplicates(subset=['family'],inplace=True)
    family_level.dropna(inplace=True)
    family_year_median_df = pd.merge(family_year_median_df, family_level, left_on='family', right_on='family')
    family_year_median_df.to_csv(r'family_year_median_df.csv')

   
    #step2
    import pandas as pd
    import numpy as np
    import pickle
    
    def save_variable(v, filename):
        f = open(filename, 'wb')
        pickle.dump(v, f)
        f.close()
        return filename

    def load_variavle(filename):
        f = open(filename, 'rb')
        r = pickle.load(f)
        f.close()
        return r


    median_df = pd.read_csv(r'family_year_median_df.csv', index_col=(0))
    print(median_df)
    family = median_df[['family', 'level']].drop_duplicates()
    family.reset_index(inplace=True)
    fam = family['family'].drop_duplicates()
    fam.to_csv(r'FamilyNum_1.csv')

   # window_year = 40
    median_array = np.array([[np.nan for year in range(window_year)] for family in range(len(family))])  # 每个物种每年的中值
    count_array = np.array([0.0 for family in range(len(family))])
    g = median_df.groupby(['family'])
    i_f = 0
    for k, single in g:
        print(i_f, len(family))
        y = single['year'].values.tolist()
        m = single['median'].values.tolist()
        count_array[i_f] = (single['count_by_family'].values.tolist())[0]
        for i in range(len(y)):
            median_array[i_f, y[i] - year1] = m[i]
        i_f += 1

    save_variable(median_array, r'medianPoint.txt')
    save_variable(count_array, r'countArray.txt')
    print(count_array)

#step3
    from statistics import median
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from collections import Counter  # 引入Counter
    import pickle
    from functools import reduce
    import seaborn as sns
    import operator
    from mpl_toolkits.mplot3d import Axes3D
    import pywt
    from tsmoothie.smoother import LowessSmoother
    from scipy import stats
    import scipy
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    from statsmodels.tsa.holtwinters import Holt


    def save_variable(v, filename):
        f = open(filename, 'wb')
        pickle.dump(v, f)
        f.close()
        return filename


    def load_variavle(filename):
        f = open(filename, 'rb')
        r = pickle.load(f)
        f.close()
        return r


    start =year1
    end = year2
    yearTotal = end - start + 1

    medianPoint = load_variavle(r'medianPoint.txt')
   
    reserve = []  
    count = 0
    loss = []
    tot = []
    for family in range(medianPoint.shape[0]):  
        if len(medianPoint[family, :10][medianPoint[family, :10] > 0]) > 0:
            if len(medianPoint[family, 10:20][medianPoint[family, 10:20] > 0]) > 0:
                if len(medianPoint[family, 20:30][medianPoint[family, 20:30] > 0]) > 0:
                    if len(medianPoint[family, 30:40][medianPoint[family, 30:40] > 0]) > 0:
                        # if len(medianPoint[family, 40:][medianPoint[family, 40:] > 0]) > 0:
                        if len(medianPoint[family, :][medianPoint[family, :] > 0]) > Time_year:
                            num = 0
                            for elem in range(yearTotal):
                                if not medianPoint[family, elem] > 0:
                                    num += 1
                                elif num:
                                    tot.append(num)
                                    num = 0
                            if num:
                                tot.append(num)
                            loss.append(
                                len(medianPoint[family, :]) - len(medianPoint[family, :][medianPoint[family, :] > 0]))
                            reserve.append(family)
                            count += 1
 
    count_array = load_variavle(r'countArray.txt')
    count = np.zeros(len(reserve))
    for i in range(len(reserve)):
        fam = reserve[i]
        count[i] = count_array[fam]

    rank = np.argsort(count)
    result = np.zeros_like(rank)
    for i in range(len(rank)):
        result[rank[i]] = i

    rank_result = list(enumerate(reserve)) 
    for i, j in rank_result:
        rank_result[i] = (result[i], j)

    save_variable(rank_result, r'reserve.txt')
    reserve = load_variavle(r'reserve.txt')

    median_df = pd.read_csv(r'family_year_median_df.csv', index_col=(0))
    print(median_df)
    family = median_df[['family', 'level']].drop_duplicates()
    family.reset_index(inplace=True)

    family['id'] = np.nan  
    for n, o in reserve:
        print(o)
        family['id'].loc[[o]] = n

    family.dropna(inplace=True)

    family.to_csv(r'vertex0.9_node_230730.csv')

    '''
    Filter the median curve
    '''
    corre = []
    def wavelet_denoising(data):
        db4 = pywt.Wavelet('db4')
        coeffs = pywt.wavedec(data, db4, level=8)
        coeffs[len(coeffs) - 2] *= 0
        coeffs[len(coeffs) - 3] *= 0
        coeffs[len(coeffs) - 4] *= 0
        meta = pywt.waverec(coeffs, db4)
        c, p = scipy.stats.pearsonr(data, meta[:len(data)])
        corre.append(c)
        return meta[:len(data)]
    median_denoising = np.array(
        [[np.nan for year in range(yearTotal)] for family in range(count_array.shape[0])])  # 每个物种每年的中值

    '''
    interpolation
    '''
    percent_up = 0
    percent_down = 0

    for family in range(count_array.shape[0]):  
        df = pd.DataFrame(medianPoint[family])
        if family in reserve:
            percent_up += len(df) - len(df.dropna())
            percent_down += len(df)
        df.fillna(df.interpolate(), inplace=True)  
        df.fillna(method='backfill', inplace=True)  
        medianPoint[family] = np.copy(np.array(df[0].values.tolist()))
    save_variable(medianPoint, r'median_interpolation.txt')

    '''
    significant
    '''
    def get_p_value(arrA, arrB):
        a = np.array(arrA)
        b = np.array(arrB)
        t, p = stats.ttest_ind(a, b)
        return p
    
    def R2_fun(y, y_forecast):
        y_mean = np.mean(y)
        return 1 - (np.sum((y_forecast - y) ** 2)) / (np.sum((y - y_mean) ** 2))

    count = 0
    for family in range(medianPoint.shape[0]):
        if family in [y for x, y in rank_result]:
            smoother = LowessSmoother(smooth_fraction=0.1, iterations=1)
            nanNum = len(medianPoint[family][np.isnan(medianPoint[family])])
            if nanNum < window_year:
                smoother.smooth(medianPoint[family][medianPoint[family] > 0])
                median_denoising[family, nanNum:] = np.copy(smoother.smooth_data[0])
            if 1 - get_p_value(medianPoint[family], median_denoising[family]) < 0.05:
                count += 1
   
    difference = []  
    for family in range(medianPoint.shape[0]):
        if family in [y for x, y in rank_result]:
            for year in range(0, yearTotal - 4):
                diff = np.max(median_denoising[family, year: year + 5]) - median_denoising[family, year]  
                difference.append(diff)
                diff = np.min(median_denoising[family, year: year + 5]) - median_denoising[family, year] 
                difference.append(diff)

    mean, std = np.mean(difference), np.std(difference, ddof=1)
    conf_intveral = stats.norm.interval(0.95, loc=mean, scale=std)
 
    count = 0

    for family in range(medianPoint.shape[0]):
        if family in [y for x, y in rank_result]:
            flag = False
            for year in range(0, yearTotal - 4):
                for time in range(4):
                    diff = median_denoising[family, year + time + 1] - median_denoising[family, year]  
                    if np.fabs(diff) > 5:
                        nanNum = len(median_denoising[family][np.isnan(median_denoising[family])])
                        if nanNum < window_year:
                            median_denoising[family, nanNum:] = np.copy(
                                wavelet_denoising(median_denoising[family][~np.isnan(median_denoising[family])]))
                            flag = True
                            break
                if flag:
                    break
            if 1 - get_p_value(medianPoint[family], median_denoising[family]) < 0.05:
                count += 1
                print([new for new, old in rank_result if old == family])

  
    count = 0
    count1 = 0
    difference = [] 
    for family in range(medianPoint.shape[0]):
        if family in [y for x, y in rank_result]:
            for year in range(0, yearTotal - 4):
                diff = np.max(median_denoising[family, year: year + 5]) - median_denoising[family, year] 
                difference.append(diff)
                diff = np.min(median_denoising[family, year: year + 5]) - median_denoising[family, year]  
                difference.append(diff)
                if np.max(median_denoising[family, year: year + 5]) - np.min(median_denoising[family, year: year + 5]) > 5:
                    count1 += 1
                count += 1
    sns.distplot(corre, kde_kws={"label": "KDE"}, vertical=False, color="y")
    save_variable(median_denoising, r'median_denoising.txt')

    #step4
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from collections import Counter #引入Counter
    import pickle
    from functools import reduce
    import seaborn as sns
    import scipy
    import operator
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

    dffamily = pd.read_csv(r'FamilyNum_1.csv')
    familyName = dffamily['family'].values.tolist()
    yearTotal = year2 - year1 + 1

    medianPoint = load_variavle(r'median_denoising.txt')
    reserve = load_variavle(r'reserve.txt')

    medianPointExtract = np.array([[0.0 for year in range(yearTotal)] for family in range(len(reserve))])
    familyExtract = [[] for family in range(len(reserve))] 

    for i, j in reserve:
        medianPointExtract[i] = np.copy(medianPoint[j])
        familyExtract[i].append(str(i) + '<new-old>' + str(j) + '-' + familyName[j])

    pvalue = np.array([[0 for col in range(len(reserve))] for row in range(len(reserve))],dtype=float)
    correlation = np.array([[0 for col in range(len(reserve))] for row in range(len(reserve))],dtype=float)
    for i in range(len(reserve)):
        for j in range(len(reserve)):
            x = medianPointExtract[i][medianPointExtract[i] > 0]
            y = medianPointExtract[j][medianPointExtract[j] > 0]
            if len(x) - len(y) > 0:
                correlation[i,j], pvalue[i,j] = scipy.stats.pearsonr(x[len(x) - len(y):], y)
            else:
                correlation[i,j], pvalue[i,j] = scipy.stats.pearsonr(x, y[len(y) - len(x):])
    save_variable(correlation,r'correlation.txt')
    save_variable(pvalue,r'pvalue.txt')
    save_variable(familyExtract,r'familyExtract.txt')

    
    #step5
    import pandas as pd
    import scipy
    from collections import Counter  
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.cluster.hierarchy as sch  
    import scipy.spatial.distance as ssd
    from scipy.cluster.vq import vq, kmeans, whiten
    import numpy as np
    import pickle


    def save_variable(v, filename):
        f = open(filename, 'wb')
        pickle.dump(v, f)
        f.close()
        return filename

    def load_variavle(filename):
        f = open(filename, 'rb')
        r = pickle.load(f)
        f.close()
        return r

    def calEuclidean(x, y):
        dist = np.sqrt(np.sum(np.square(x - y)))  
        return dist

    medianPoint = load_variavle(r'median_denoising.txt')  # median
    reserve = load_variavle(r'reserve.txt')  # new - old
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
    Z = sch.linkage(disMat, method='average')
    cluster = sch.fcluster(Z, t=15, criterion='distance')
    l = Counter(cluster).most_common(15)
  
    save_variable(cluster, r'cluster.txt')  
    np.savetxt(r'label_230730.csv', cluster, fmt='%d', delimiter=',')
    plt.show()

    for i in range(len(cluster)):
        if cluster[i] == 3:
            idx = [o for n, o in reserve if n == i][0]
            plt.plot(medianPoint[idx])
    plt.show()

 
    #step6
    from statistics import median
    from turtle import position
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from collections import Counter  # 引入Counter
    import pickle
    import seaborn as sns
    import scipy
    import operator
    from queue import Queue
    import networkx as nx
    import pickle


    def save_variable(v, filename):
        f = open(filename, 'wb')
        pickle.dump(v, f)
        f.close()
        return filename


    def load_variavle(filename):
        f = open(filename, 'rb')
        r = pickle.load(f)
        f.close()
        return r


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
    familyExtract = load_variavle(r'familyExtract.txt')  # family name
    medianPoint = load_variavle(r'median_denoising.txt')  # median
    print(medianPoint.shape[0])
    reserve = load_variavle(r'reserve.txt')  # new - old
   
    wholegroup = []
    wholeedgelist = []


    for fam in range(medianPoint.shape[0]):
        if fam == correlation.shape[0]: 
            break
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
 
    #step7
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Polygon  
    from matplotlib.figure import Figure
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from matplotlib.collections import PatchCollection
    from matplotlib.transforms import TransformedBbox, Bbox
    from matplotlib.image import BboxImage
    from matplotlib.legend_handler import HandlerBase
    from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
    import os
    from mk_test import mk_test
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
    node = [] # 非孤立节点
    df['source'] = df['source'].astype(int)
    df['target'] = df['target'].astype(int)

    for elem in df['source']:
        if not elem in node:
            node.append(elem)

    for elem in df['target']:
        if not elem in node:
            node.append(elem)

    reserve = load_variavle(r'reserve.txt')
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
    total = [df[df['mode'] == 0], df[df['mode'] == 1],
             df[df['mode'] == 2], df[df['mode'] == 3]]
    data_percentage=[]
    for idx in range(4):
        level = total[idx]['level'].values.tolist()
        # num = total[idx]['family_num'].values.tolist()
        level_prop = [len([i for i in level if i == 1]) / len(level),
                      len([i for i in level if i == 2]) / len(level),
                      len([i for i in level if i == 3]) / len(level),
                      len([i for i in level if i == 4]) / len(level),
                      len([i for i in level if i == 5]) / len(level)]

        level_prop = [i * 1.5 for i in level_prop]

        #print(level_prop)
        data_percentage.append(level_prop)

        data_left = [idx + 1 - i/2 for i in level_prop]
        data_right = [idx + 1 + i/2 for i in level_prop]
        color_list = [(140/255, 185/255, 0/255), (20/255, 178/255, 255/255), (255/255, 194/255, 102/255),
                      (255/255, 112/255, 69/255), (255/255, 26/255, 128/255)]  # 柱子颜色

        axes = []
        for l in range(5):
            axes.append(ax.barh(y=l + 1, width=level_prop[l], left=data_left[l],
                        color=color_list[l], height=0.9, label='{}'.format(l + 1)))  # 柱宽设置为0.7
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
            lg.set_title(title='Trophic level', prop = font1)
        polygons = []
        for i in range(5):
            ax.text(
                idx + 1,  
                i + 1, 
                "%.2f" % (level_prop[i] / 1.5 * 100) + '%',  # 文本
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


    set_axis(ax)
    ax.set_xlim([0.5, 4.5])
    ax.set_ylim([0.3, 5.7])
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(['Total', 'Northward', 'Southward', 'Mixed'], font1)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels([1, 2, 3, 4, 5], font1)
    ax.text(0.6, 0.3 + 5.4 * 0.95, 'B', font3)
    fig = plt.gcf()
    fig.set_size_inches(8, 8)
    suffix1='.pdf'
    suffix2 = '.png'
    filename1= str(year2)+suffix1
    filename2= str(year2) + suffix2
    fig.savefig(filename1, dpi=500)
    fig.savefig(filename2, dpi=800)
    plt.show()
    data_end.append(data_percentage)
    
    import os
    os.remove('correlation.txt')
    os.remove('countArray.txt')
    os.remove('cluster.txt')
    os.remove('familyExtract.txt')
    os.remove('FamilyNum_1.csv')
    os.remove('label_230730.csv')
    os.remove('median_denoising.txt')
    os.remove('median_interpolation.txt')
    os.remove('medianPoint.txt')
    os.remove('pvalue.txt')
    os.remove('reserve.txt')
    os.remove('vertex0.9_cluster_230730.csv')
    os.remove('vertex0.9_node_230730.csv')
    os.remove('family_year_median_df.csv')

flat_data = [item for sublist in data_end for item in sublist]

import numpy as np
np.savetxt(r'data_end.txt',flat_data)