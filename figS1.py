import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from collections import Counter #引入Counter
import pickle
from functools import reduce
import seaborn as sns
import scipy
from scipy import stats

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

font = {
        'weight': 'bold',
        'style':'normal',
        'size': 15,
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
    ax.set_xticks([0,200,400,600,800,1000,1200,1400])
    ax.set_xticklabels(ax.get_xticks(),font)
    ax.set_ylabel('Record Number for Each Family',font)
    ax.set_yticks([int(1e1),int(1e2),int(1e3),int(1e4),int(1e5),int(1e6)])
    ax.set_yticklabels(['10$^1$','10$^2$','10$^3$','10$^4$','10$^5$','10$^6$'],font)
    ax.set_xlabel('Family Index',font)

reserve = load_variavle(r'chugao\fish_220114_re\correlation_median_point\reserve.txt')
latbyYear= load_variavle(r'chugao\fish_220114_re\cluster\latitudeByYear_1970_2020_2846_1.txt')
record = np.array(latbyYear)

quantity = np.array([0 for fam in range(len(reserve))])

for fam in range(len(reserve)):
    old = [o for n, o in reserve if n == fam][0]
    quantity[fam] = np.sum([len(record[year, old, 0]) for year in range(0,51)])

ax = plt.subplot(111)
ax.set_yscale("log")
ax.bar(np.arange(0,len(reserve),1), quantity)
set_axis(ax)

fig = plt.gcf()
fig.set_size_inches(10, 5)
fig.savefig(r'chugao\fish_220114_re\SM\S1.png',dpi=1000)
plt.show()