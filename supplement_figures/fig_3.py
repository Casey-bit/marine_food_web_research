'''
 # Copyright (C) 2022-08-09 School of IoT, Jiangnan University Limited
 #
 # All Rights Reserved.
 #
 # Paper: Cascade Shifts of Marine Species in the Northern Hemisphere
 # First author: Zhenkai Wu
 # Corresponding author: Ya Guo, E-mail: guoy@jiangnan.edu.cn
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
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


def get_p_value(arrA, arrB):
    a = np.array(arrA)
    b = np.array(arrB)
    t, p = stats.ttest_ind(a,b)
    return p

def R2_fun(y, y_forecast):
    y_mean=np.mean(y)
    return 1 - (np.sum((y_forecast - y) ** 2)) / (np.sum((y - y_mean) ** 2))

import matplotlib.colors as colors_tmp


class MidpointNormalize(colors_tmp.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        colors_tmp.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
font1 = {
        'weight': 'bold',
        'style':'normal',
        'size': 15,
        }
font2 = {
        'weight': 'bold',
        'style':'normal',
        'size': 12,
        }

def set_axis(n,ax):
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
    if n == 0:
        ax.set_yticks([0,10,20,30,40,50,60,70])
        ax.set_yticklabels(ax.get_yticks(),font1)
        ax.set_ylabel('Latitude (°N)',font1)
    else:
        ax.set_yticklabels(ax.get_yticklabels(),font1)
    ax.set_xticks([1970,1980,1990,2000,2010,2020])
    ax.set_xticklabels(ax.get_xticks(),font1)
    ax.set_xlabel('Year',font1)
    

cluster = load_variavle(r'chugao\fish_220114_re\figure3\cluster.txt') # 559 所属类

reserve = load_variavle(r'chugao\fish_220114_re\correlation_median_point\reserve559.txt') # new - old
medianPoint = load_variavle(r'chugao\fish_220114_re\correlation_median_point\median_denoising.txt') # median 2846,51

font = {
        'weight': 'bold',
        'style':'normal',
        'size': 20,
}
for item in range(4):

    if item < 3:
        ax = plt.subplot2grid((15,3),(0,item),rowspan=13,colspan=1)
    else:
        cax = plt.subplot2grid((15,3),(14,0),rowspan=1,colspan=3)
        # cax.set_xticklabels(['1.13×10$^{-4}$','3.54×10$^{-4}$','6.26×10$^{-4}$','8.95×10$^{-4}$','1.16×10$^{-3}$','1.40×10$^{-3}$','1.62×10$^{-3}$','1.81×10$^{-3}$','1.97×10$^{-3}$','2.14×10$^{-3}$'],font1)
        # cax.set_xlabel('Family Density: Distribution density of families in the total 559 species,  Latitude resolution and 1 Year resolution',font1)
        plt.figure(2)
        ax = plt.subplot2grid((15,3),(14,0),rowspan=1,colspan=3)
    ax.set_xlim([1967,2023])
    ax.set_ylim([0,70])
    kdex = []
    kdey = []
    count = 0

    for fam in range(559):
        if item == 2:
            if not cluster[fam] == 3 and not cluster[fam] == 14:
                old = [o for n, o in reserve if n == fam][0]
                kdex += [i for i in range(1970,2021,1)]
                kdey += [i for i in medianPoint[old]]
                count += 1
                ax.text(1971.5, 68, 'C Mixed', verticalalignment="top",
                            horizontalalignment="left", fontdict=font)
        elif item == 3:
            if not cluster[fam] == 3 and not cluster[fam] == 14:
                old = [o for n, o in reserve if n == fam][0]
                kdex += [i for i in range(1970,2021,1)]
                kdey += [i for i in medianPoint[old]]
                count += 1
                ax.text(1971.5, 68, 'C Mixed', verticalalignment="top",
                            horizontalalignment="left", fontdict=font)

        elif item == 0:
            if cluster[fam] == 3:
                old = [o for n, o in reserve if n == fam][0]
                kdex += [i for i in range(1970,2021,1)]
                kdey += [i for i in medianPoint[old]]
                count += 1
                ax.text(1971.5, 68, 'A Northward', verticalalignment="top",
                            horizontalalignment="left", fontdict=font)
        elif item == 1:
            if cluster[fam] == 14:
                old = [o for n, o in reserve if n == fam][0]
                kdex += [i for i in range(1970,2021,1)]
                kdey += [i for i in medianPoint[old]]
                count += 1
                ax.text(1971.5, 68, 'B Southward', verticalalignment="top",
                            horizontalalignment="left", fontdict=font)
    set_axis(item,ax)
    # plt.subplot(1,2,1)
    # plt.scatter(kdex,kdey)
    # plt.subplot(1,2,2)
    # ax = plt.subplot(1,3,2)

    # midnorm = MidpointNormalize(vmin=0, vcenter=0.0025, vmax=0.005)  
    midnorm = colors_tmp.TwoSlopeNorm(vmin = 0, vcenter = 0.0015, vmax = 0.003)
    # ax.mappable.set_norm(midnorm)
    my_kde = sns.kdeplot(kdex,kdey,shade=True,levels = 10,bw=0.5,cmap='cool',)
    if item == 3:
        my_kde = sns.kdeplot(kdex,kdey,shade=True,cbar=True,levels = 10,bw=0.5,cmap='cool',
                              cbar_ax = cax,cbar_kws = {'label': 'Family Density','orientation':'horizontal','format':'%.2e','pad':0.015,'norm':midnorm})
        cax.set_xticklabels(['1.13×10$^{-4}$','3.54×10$^{-4}$','6.26×10$^{-4}$','8.95×10$^{-4}$','1.16×10$^{-3}$','1.40×10$^{-3}$','1.62×10$^{-3}$','1.81×10$^{-3}$','1.97×10$^{-3}$','2.14×10$^{-3}$'],font1)
        cax.set_xlabel('Family Density: Distribution density of families in the total 559 families, 0.5° Latitude resolution and 1 Year resolution',font1)
 

    para = np.polyfit(kdex,kdey,1)

    x = np.arange(1970,2021,1)
    y2 = list(para[0] * x **1 + para[1])

    ax.plot(x,y2,'r',label='Linear fitting (p < 0.001)')
    ax.legend(prop = font2,loc=3)
    y22 = []
    beishu = int(len(kdex) / len(x))
    for i in range(beishu):
        y22 += y2
    print(len(y22),len(kdey))
    r2 = R2_fun(np.array(kdey), np.array(y22))
    p = get_p_value(np.array(kdey), np.array(y22))
    print(r2, p)

plt.figure(1)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, 
    wspace=0.1, hspace=0.1)
fig = plt.gcf()
fig.set_size_inches(18, 10)
fig.savefig(r'chugao\fish_220114_re\SM\figS3.png',dpi=500)
plt.show()
