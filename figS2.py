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
font1 = {
        'weight': 'bold',
        'style':'normal',
        'size': 25,
        }
font2 = {
        'weight': 'bold',
        'style':'normal',
        'size': 12,
        }

def set_axis(ax):
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
    ax.set_yticks([0,200,400,600,800])
    ax.set_yticklabels(ax.get_yticks(),font1)
    ax.set_ylabel('Number of Families in Migration',font1)
    # ax.set_xticks([1970,1980,1990,2000,2010,2020])
    ax.set_xticklabels(['Northward','Southward'],font1)
    # ax.set_xlabel('Year',font1)
    ax.set_xlim(-0.5,1.5)
medianPoint = load_variavle(r'chugao\fish_220114_re\correlation_median_point\medianPoint.txt') # 2846,51
print(medianPoint.shape)
reserve = load_variavle(r'chugao\fish_220114_re\correlation_median_point\reserve.txt') # new - old


north_1 = 0
south_1 = 0

for fam in range(2846):
    new = [n for n, o in reserve if o == fam]
    if len(new) > 0:
        lat1 = medianPoint[fam,25:]
        lat2 = medianPoint[fam,:25]
        if np.mean([i for i in lat1 if i > 0]) >= np.mean([i for i in lat2 if i > 0]):
            north_1 += 1
        else:
            south_1 += 1
print(north_1,south_1)
ax = plt.subplot(111)
ax.bar(['Northward','Southward'],[north_1,south_1],width = 0.3)
set_axis(ax)
fig = plt.gcf()
fig.set_size_inches(10, 10)
fig.savefig(r'chugao\fish_220114_re\SM\S2.png',dpi=1000)
plt.show()