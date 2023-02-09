import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
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

dffamily = pd.read_csv(r'fig2\latitude\FamilyNum_1.csv')
familyName = dffamily['family'].values.tolist()
yearTotal = 2020 - 1970 + 1
'''
按照reserve里面的顺序提取出medianPoint中的物种
'''
# 2841,51
medianPoint = load_variavle(r'fig2\latitude\median_denoising.txt')
reserve = load_variavle(r'fig2\latitude\reserve.txt')

medianPointExtract = np.array([[0.0 for year in range(yearTotal)] for family in range(len(reserve))])
familyExtract = [[] for family in range(len(reserve))] # 提取出的family名，按照reserve顺序排列

for i, j in reserve:
    medianPointExtract[i] = np.copy(medianPoint[j])
    familyExtract[i].append(str(i) + '<new-old>' + str(j) + '-' + familyName[j])

# plt.plot(np.arange(0,len(medianPointExtract[270]),1), medianPointExtract[270],label = familyExtract[270][0])
# plt.plot(np.arange(0,len(medianPointExtract[711]),1), medianPointExtract[711],label = familyExtract[711][0])
# plt.legend()
# plt.show()

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
# correlation = np.fabs(correlation)
save_variable(correlation,r'fig2\latitude\correlation.txt')
save_variable(pvalue,r'fig2\latitude\pvalue.txt')
save_variable(familyExtract,r'fig2\latitude\familyExtract.txt')

# df_r = pd.DataFrame({})
# df_r['Family'] = familyExtract
# for o in range(len(familyExtract)):
#     df_r[familyExtract[o]] = correlation[:, o]
# df_r.to_csv(r'chugao\fish_220114_re\correlation_median_point\correlation.csv')

# df_p = pd.DataFrame({})
# df_p['Family'] = familyExtract
# for o in range(len(familyExtract)):
#     df_p[familyExtract[o]] = pvalue[:, o]
# df_p.to_csv(r'chugao\fish_220114_re\correlation_median_point\pvalue.csv')