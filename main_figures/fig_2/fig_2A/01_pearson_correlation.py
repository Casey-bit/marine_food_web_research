import numpy as np
import pandas as pd
import pickle
import scipy
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

dffamily = pd.read_csv('E:\\paper\\obis_20220114.csv\\familyNum.csv')
familyName = dffamily['family'].values.tolist()
yearTotal = 2020 - 1970 + 1
'''
extract families accoring to the reserve_family.txt
'''
# 2841,51
medianPoint = load_variavle(r'chugao\fish_220114_re\correlation_median_point\median_denoising.txt')
reserve = load_variavle(r'chugao\fish_220114_re\correlation_median_point\reserve_family.txt')

medianPointExtract = np.array([[0.0 for year in range(yearTotal)] for family in range(len(reserve))])
familyExtract = [[] for family in range(len(reserve))] # 提取出的family名，按照reserve顺序排列

for i, j in reserve:
    medianPointExtract[i] = np.copy(medianPoint[j])
    familyExtract[i].append(str(i) + '<new-old>' + str(j) + '-' + familyName[j])


pvalue = np.array([[0 for col in range(len(reserve))] for row in range(len(reserve))],dtype=float)
correlation = np.array([[0 for col in range(len(reserve))] for row in range(len(reserve))],dtype=float)
for i in range(len(reserve)):
    for j in range(len(reserve)):
        x = medianPointExtract[i][medianPointExtract[i] > 0]
        # x = x[2:-2]
        y = medianPointExtract[j][medianPointExtract[j] > 0]
        # y = y[2:-2]
        if len(x) - len(y) > 0:
            correlation[i,j], pvalue[i,j] = scipy.stats.pearsonr(x[len(x) - len(y):], y)
        else:
            correlation[i,j], pvalue[i,j] = scipy.stats.pearsonr(x, y[len(y) - len(x):])
    print(i)

save_variable(correlation,r'chugao\fish_220114_re\correlation_median_point\correlation.txt')
save_variable(pvalue,r'chugao\fish_220114_re\correlation_median_point\pvalue.txt')
save_variable(familyExtract,r'chugao\fish_220114_re\correlation_median_point\familyExtract.txt')
