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
from mpl_toolkits.mplot3d import Axes3D
sns.set_style("whitegrid")

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
    ax.tick_params(which="major",
                    length=15,width = 2.0,
                    colors = "black",
                    direction = 'in',
                    tick2On = False)

reserve = load_variavle(r'chugao\fish_220114_re\correlation_median_point\reserve.txt')

medianPoint = load_variavle(r'chugao\fish_220114_re\correlation_median_point\medianPoint.txt')

medianDenoising = load_variavle(r'chugao\fish_220114_re\correlation_median_point\median_denoising.txt')


font1 = {
        'weight': 'bold',
        'style':'normal',
        'size': 30,
        }
font2 = {
        'weight': 'bold',
        'style':'normal',
        'size': 20,
        }

ax = plt.subplot(111)
for i in range(8,9):
    old = [o for n, o in reserve if n == i][0]
    ax.plot(np.arange(1970,2021,1), medianPoint[old], linewidth = 3, label = 'One specific initial family shift trajectory')
    ax.plot(np.arange(1970,2021,1), medianDenoising[old], linewidth = 3, label = 'The corresponding trajectory after denoising')
    set_axis(ax)
    ax.set_xticks([1970,1980,1990,2000,2010,2020])
    ax.set_xticklabels([1970,1980,1990,2000,2010,2020],font1)
    ax.set_yticks([30,40,50,60,70])
    ax.set_ylabel('Latitude (°N)',font1)
    ax.set_yticklabels([30,40,50,60,70],font1)
    ax.set_xlabel('Year',font1)
    # ax.set_xlim([1972,2017])
    ax.set_ylim([25,75])
    ax.legend(prop = font2,ncol = 1,loc = 2)
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    fig.savefig(r'chugao\fish_220114_re\SM\S5.png',dpi=1000)
    plt.show()