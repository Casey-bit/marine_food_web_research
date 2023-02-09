from statistics import median
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from collections import Counter #引入Counter
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

start = 1970
end = 2020
yearTotal = end - start + 1


medianPoint = load_variavle(r'fig2\latitude\medianPoint.txt')
'''
计算需要保留下来的物种
'''
reserve = [] # 记录保留的物种
count = 0
loss = []
tot = []
for family in range(811):
    if len(medianPoint[family, :10][medianPoint[family, :10] > 0]) > 0:
        if len(medianPoint[family, 10:20][medianPoint[family, 10:20] > 0]) > 0:
            if len(medianPoint[family, 20:30][medianPoint[family, 20:30] > 0]) > 0:
                if len(medianPoint[family, 30:40][medianPoint[family, 30:40] > 0]) > 0:
                    if len(medianPoint[family, 40:][medianPoint[family, 40:] > 0]) > 0:
                        if len(medianPoint[family, :][medianPoint[family, :] > 0]) > 35:
                            # 如果51年中有35年有数据，就保留下来
                            num = 0
                            for elem in range(51):
                                if not medianPoint[family, elem] > 0:
                                    num += 1
                                elif num:
                                    tot.append(num)
                                    num = 0
                            if num:
                                tot.append(num)
                            loss.append(len(medianPoint[family, :]) - len(medianPoint[family, :][medianPoint[family, :] > 0]))
                            reserve.append(family)
                            count += 1
# print(len([i for i in tot if i <= 5]) / len(tot))
# plt.boxplot(tot,showfliers=False)
# # sns.distplot(tot,kde_kws={"label":"KDE"},vertical=False,color="y")
# plt.show()

count_array = load_variavle(r'fig2\latitude\countArray.txt')
count = np.zeros(len(reserve))
for i in range(len(reserve)):
    fam = reserve[i]
    count[i] = count_array[fam]

rank = np.argsort(count)
result = np.zeros_like(rank)
for i in range(len(rank)):
    result[rank[i]] = i

rank_result = list(enumerate(reserve)) # 重新排序后的顺序，new order --- old order
for i, j in rank_result:
    rank_result[i] = (result[i], j)

print(rank_result)
print(len(rank_result))

save_variable(rank_result,r'fig2\latitude\reserve.txt')


'''
对median曲线滤噪
'''

corre = []

# 小波滤噪
def wavelet_denoising(data):
    # 小波函数取db4
    db4 = pywt.Wavelet('db4')
    # 分解
    coeffs = pywt.wavedec(data, db4, level=8)
    # 高频系数置零
    
    # plt.subplot(6 ,2, 1)
    # plt.title('initial')
    # plt.plot(np.arange(0,len(data),1),data)
    # for i in range(len(coeffs)):
    #     extract = []
    #     for j in range(len(coeffs)):
    #         if j == i:
    #             extract.append(coeffs[j])
    #         else:
    #             extract.append(coeffs[j] * 0)
    #     test = pywt.waverec(extract, db4)
    #     plt.subplot(6,2,i + 3)
    #     plt.title(i)
    #     plt.plot(np.arange(0,len(test),1),test)

    # coeffs[len(coeffs)-1] *= 0
    coeffs[len(coeffs)-2] *= 0
    coeffs[len(coeffs)-3] *= 0
    coeffs[len(coeffs)-4] *= 0
    # coeffs[len(coeffs)-5] *= 0
    # coeffs[len(coeffs)-6] *= 0
    # 重构
    meta = pywt.waverec(coeffs, db4)
    # plt.subplot(6 ,2, 2)
    # plt.title('final')
    # plt.plot(np.arange(0,len(meta),1),meta)
    # print(scipy.stats.pearsonr(data, meta[:len(data)]))
    # plt.show()
    c, p = scipy.stats.pearsonr(data, meta[:len(data)])
    corre.append(c)

    return meta[:len(data)]

median_denoising = np.array([[np.nan for year in range(yearTotal)] for family in range(811)]) # 每个物种每年的中值

'''
插值
'''
for family in range(811):
    df = pd.DataFrame(medianPoint[family])
    df.fillna(df.interpolate(),inplace=True) # 最近前后平均
    df.fillna(method='backfill',inplace=True) # 向前插值，弥补最开始的缺失值
    medianPoint[family] = np.copy(np.array(df[0].values.tolist()))

save_variable(medianPoint, r'fig2\latitude\median_interpolation.txt')

'''
显著性
'''

def get_p_value(arrA, arrB):
    a = np.array(arrA)
    b = np.array(arrB)
    t, p = stats.ttest_ind(a,b)
    return p

def R2_fun(y, y_forecast):
    # 拟合优度R^2
    y_mean=np.mean(y)
    return 1 - (np.sum((y_forecast - y) ** 2)) / (np.sum((y - y_mean) ** 2))

'''
局部加权回归散点图平滑
'''

count = 0

for family in range(medianPoint.shape[0]):
    if family in [y for x, y in rank_result]:
        smoother = LowessSmoother(smooth_fraction=0.1, iterations=1)
        nanNum = len(medianPoint[family][np.isnan(medianPoint[family])])
        if nanNum < 51:
            smoother.smooth(medianPoint[family][medianPoint[family] > 0])
            median_denoising[family, nanNum:] = np.copy(smoother.smooth_data[0])
        if 1 - get_p_value(medianPoint[family], median_denoising[family]) < 0.05:
            count += 1
print(count / len(rank_result))
        # print(family)

'''

'''

difference = [] # 五年差
for family in range(medianPoint.shape[0]):
    if family in [y for x, y in rank_result]:
        for year in range(0, yearTotal - 4):
            diff = np.max(median_denoising[family, year: year + 5]) - median_denoising[family, year] # 最北差值            
            difference.append(diff)
            diff = np.min(median_denoising[family, year: year + 5]) - median_denoising[family, year] # 最南差值            
            difference.append(diff)

mean, std = np.mean(difference), np.std(difference,ddof=1)
print("kstest",stats.kstest(difference, 'norm',(mean,std)))

# 计算置信区间
conf_intveral = stats.norm.interval(0.95, loc=mean, scale=std)
print(conf_intveral)
print(np.percentile(difference,2.5),np.percentile(difference,97.5))

sns.distplot(difference,kde_kws={"label":"KDE"},vertical=False,color="y")
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
# plt.xlim([0,10])
# plt.ylim([0,3])
plt.grid()
plt.show()



count = 0

for family in range(medianPoint.shape[0]):
    if family in [y for x, y in rank_result]:
        flag = False
        for year in range(0, yearTotal - 4):
            for time in range(4):
                diff = median_denoising[family, year + time + 1] - median_denoising[family, year] # 差值
                if np.fabs(diff) > 5:
                    # median_denoising[family, year + time + 1] = median_denoising[family, year] + diff / 2
                    nanNum = len(median_denoising[family][np.isnan(median_denoising[family])])
                    if nanNum < 51:
                        # ets1 = Holt(median_denoising[family], damped_trend=True)
                        # r1 = ets1.fit(smoothing_level=0.1, smoothing_slope=0.01)
                        # # median_denoising[family, nanNum:] = np.copy(r1.fittedvalues)
                        # median_denoising[family, nanNum:] = np.copy(scipy.signal.savgol_filter(median_denoising[family], 21, 2))
                        
                        median_denoising[family, nanNum:] = np.copy(wavelet_denoising(median_denoising[family][~np.isnan(median_denoising[family])]))

                        flag = True
                        break
            if flag:
                break
        if 1 - get_p_value(medianPoint[family], median_denoising[family]) < 0.05:
            count += 1
            print([new for new,old in rank_result if old == family])
print(count / len(rank_result))

'''
'''
count = 0
count1 = 0
difference = [] # 五年差
for family in range(medianPoint.shape[0]):
    if family in [y for x, y in rank_result]:
        for year in range(0, yearTotal - 4):
            diff = np.max(median_denoising[family, year: year + 5]) - median_denoising[family, year] # 最北差值            
            difference.append(diff)
            diff = np.min(median_denoising[family, year: year + 5]) - median_denoising[family, year] # 最南差值            
            difference.append(diff)
            if np.max(median_denoising[family, year: year + 5]) - np.min(median_denoising[family, year: year + 5]) > 5:
                count1 += 1
            count += 1
print('===' ,(count - count1) / count)
print(np.percentile(difference,2.5),np.percentile(difference,97.5))
sns.distplot(difference,kde_kws={"label":"KDE"},vertical=False,color="y")
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
# plt.xlim([-1.25,1.25])
# plt.ylim([0,3])
plt.grid()
plt.show()


sns.distplot(corre,kde_kws={"label":"KDE"},vertical=False,color="y")
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.xlim([-1.25,1.25])
plt.ylim([0,3])
plt.grid()
plt.title('delete2345 correlation_mean:' + str(np.mean(corre)),fontsize = 20)
plt.show()

save_variable(median_denoising, r'fig2\latitude\median_denoising.txt')