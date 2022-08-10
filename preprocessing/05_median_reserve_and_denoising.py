from statistics import median
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
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

start = 1970
end = 2020
yearTotal = end - start + 1

latitudeByYear = load_variavle(r'chugao\fish_220114_re\cluster\latitudeByYear_1970_2020_2846_family.txt')

familynum = 2846

'''
calculate latitudinal medians for each family in each year
'''
medianPoint = np.array([[0.0 for year in range(yearTotal)] for family in range(familynum)])
record = np.array(latitudeByYear)
for year in range(yearTotal):
    for family in range(familynum):
        if len(record[year, family, 0]) > 0:
            medianPoint[family, year] = median(record[year, family, 0])
        else:
             medianPoint[family, year] = np.nan

save_variable(medianPoint, r'chugao\fish_220114_re\correlation_median_point\medianPoint.txt')

'''
determine the reserved families
'''
reserve = [] # the reserved families
count = 0
loss = []
tot = []
for family in range(familynum):
    if len(medianPoint[family, :10][medianPoint[family, :10] > 0]) > 0:
        # if len(medianPoint[family, 10:20][medianPoint[family, 10:20] > 0]) > 0:
        #     if len(medianPoint[family, 20:30][medianPoint[family, 20:30] > 0]) > 0:
        #         if len(medianPoint[family, 30:40][medianPoint[family, 30:40] > 0]) > 0:
        #             if len(medianPoint[family, 40:][medianPoint[family, 40:] > 0]) > 0:
        #                 if len(medianPoint[family, :][medianPoint[family, :] > 0]) > 35:
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

count = np.zeros(len(reserve))
for i in range(len(reserve)):
    fam = reserve[i]
    for year in range(yearTotal):
        count[i] -= len(record[year, fam, 0])

# sort the reserved families by records number
rank = np.argsort(count)
result = np.zeros_like(rank)
for i in range(len(rank)):
    result[rank[i]] = i

rank_result = list(enumerate(reserve)) # the sorted pairs : new index --- old index
for i, j in rank_result:
    rank_result[i] = (result[i], j)

print(rank_result)
print(len(rank_result))

save_variable(rank_result,r'chugao\fish_220114_re\correlation_median_point\reserve_family.txt')


'''
denoising for the median curves
'''

corre = []

# wavelet denoising
def wavelet_denoising(data):
    # wavelet function : db4
    db4 = pywt.Wavelet('db4')

    # deconstruction
    coeffs = pywt.wavedec(data, db4, level=8)
    # set high frequency coefficient to zero

    # coeffs[len(coeffs)-1] *= 0
    coeffs[len(coeffs)-2] *= 0
    coeffs[len(coeffs)-3] *= 0
    coeffs[len(coeffs)-4] *= 0
    # coeffs[len(coeffs)-5] *= 0
    # coeffs[len(coeffs)-6] *= 0

    # reconstruction
    meta = pywt.waverec(coeffs, db4)
    c, p = scipy.stats.pearsonr(data, meta[:len(data)])
    corre.append(c)

    return meta[:len(data)]

median_denoising = np.array([[np.nan for year in range(yearTotal)] for family in range(familynum)]) # 每个物种每年的中值

'''
interpolation
'''
for family in range(familynum):
    df = pd.DataFrame(medianPoint[family])
    df.fillna(df.interpolate(),inplace=True) # 最近前后平均
    df.fillna(method='backfill',inplace=True) # 向前插值，弥补最开始的缺失值
    medianPoint[family] = np.copy(np.array(df[0].values.tolist()))

save_variable(medianPoint, r'chugao\fish_220114_re\correlation_median_point\median_interpolation.txt')

'''
significance
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
LOWESS regression
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

'''

'''

difference = [] # 五年差
for family in range(medianPoint.shape[0]):
    if family in [y for x, y in rank_result]:
        for year in range(0, yearTotal - 4):
            diff = np.max(median_denoising[family, year: year + 5]) - median_denoising[family, year]            
            difference.append(diff)
            diff = np.min(median_denoising[family, year: year + 5]) - median_denoising[family, year]            
            difference.append(diff)

mean, std = np.mean(difference), np.std(difference,ddof=1)
print("kstest",stats.kstest(difference, 'norm',(mean,std)))

# calculate confidence intveral
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
                diff = median_denoising[family, year + time + 1] - median_denoising[family, year]
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
difference = [] # difference in 5 years
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

save_variable(median_denoising, r'chugao\fish_220114_re\correlation_median_point\median_denoising.txt')