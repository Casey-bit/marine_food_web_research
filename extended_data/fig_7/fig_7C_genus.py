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
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from collections import Counter #引入Counter
import pickle
from functools import reduce
import seaborn as sns
import scipy
from scipy import stats
import operator
from mpl_toolkits.mplot3d import Axes3D
from tsmoothie.smoother import LowessSmoother

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

def three(ax_three):

    start = 1970
    end = 2020
    yearTotal = end - start + 1

    # year family range
    percent = load_variavle(r'chugao\fish_220114_re\cluster\percent_genus_3056.txt')
    reserve = load_variavle(r'chugao\fish_220114_re\correlation_median_point\reserve_genus.txt')
    countByYear = load_variavle('chugao\\fish_220114_re\\countByYear_1degree_genus_1970_2020_3056_1.txt')
    record = np.array(countByYear)

    percentUp = [[[] for ran in range(3)] for year in range(yearTotal - 1)]
    percentDown = [[[] for ran in range(3)] for year in range(yearTotal - 1)] # 记录三个区间每年比例上升下降的物种
    familyUp = [[[] for ran in range(3)] for year in range(yearTotal - 1)]
    familyDown = [[[] for ran in range(3)] for year in range(yearTotal - 1)]

    for year in range(yearTotal - 1):
        total = np.sum([record[year: year + 2, old, 90:] for new, old in reserve])
        print(year,total)
        for family in range(3056):
            if family in [old for new,old in reserve]:
                newfamily = [new for new,old in reserve if old == family][0]
                for ran in range(3):
                    if percent[year + 1, newfamily, ran] == np.max(percent[year + 1, newfamily]) and not percent[year, newfamily, ran] == np.max(percent[year, newfamily]):
                        # 这个区间多的物种
                        percentUp[year][ran].append(np.sum(record[year: year + 2, family, 90:]) / total)
                        familyUp[year][ran].append(newfamily)
                    if percent[year, newfamily, ran] == np.max(percent[year, newfamily]) and not percent[year + 1, newfamily, ran] == np.max(percent[year + 1, newfamily]):
                        # 这个区间少的物种
                        percentDown[year][ran].append(np.sum(record[year: year + 2, family, 90:]) / total)
                        familyDown[year][ran].append(newfamily)

    aveFamilyUp = np.array([[0.0 for ran in range(3)] for year in range(yearTotal - 1)])
    aveFamilyDown = np.array([[0.0 for ran in range(3)] for year in range(yearTotal - 1)])
    aveFamilyUpNum = np.array([[0.0 for ran in range(3)] for year in range(yearTotal - 1)])
    aveFamilyDownNum = np.array([[0.0 for ran in range(3)] for year in range(yearTotal - 1)])

    for year in range(yearTotal - 1):
        for ran in range(3):
            aveFamilyUp[year, ran] = np.mean(familyUp[year][ran])
            aveFamilyDown[year, ran] = np.mean(familyDown[year][ran])
            aveFamilyUpNum[year, ran] = np.sum(percentUp[year][ran])
            aveFamilyDownNum[year, ran] = np.sum(percentDown[year][ran])

    # save_variable(aveFamilyUp,r'chugao\fish_220114_re\cluster\aveFamilyUpfour.txt')
    # save_variable(aveFamilyDown,r'chugao\fish_220114_re\cluster\aveFamilyDownfour.txt')
    # save_variable(aveFamilyUpNum,r'chugao\fish_220114_re\cluster\aveFamilyUpNum.txt')
    # save_variable(aveFamilyDownNum,r'chugao\fish_220114_re\cluster\aveFamilyDownNum.txt')

    # aveFamilyUpNum = load_variavle(r'chugao\fish_220114_re\cluster\aveFamilyUpNum.txt')
    # aveFamilyDownNum = load_variavle(r'chugao\fish_220114_re\cluster\aveFamilyDownNum.txt')

    # 滑动
    aveUp = np.array([[0.0 for ran in range(3)] for year in range(yearTotal - 12)])
    aveDown = np.array([[0.0 for ran in range(3)] for year in range(yearTotal - 12)])
    for year in range(yearTotal - 12):
        for ran in range(3):
            # aveUp[year, ran] = np.mean(aveFamilyUp[year: year + 10, ran][aveFamilyUp[year: year + 10, ran] > 0])
            # aveDown[year, ran] = np.mean(aveFamilyDown[year: year + 10, ran][aveFamilyDown[year: year + 10, ran] > 0])

            aveUp[year, ran] = np.mean(aveFamilyUpNum[year: year + 10, ran][aveFamilyUpNum[year: year + 10, ran] > 0])
            aveDown[year, ran] = np.mean(aveFamilyDownNum[year: year + 10, ran][aveFamilyUpNum[year: year + 10, ran] > 0])


    colors = [(236/255,95/255,116/255,1),(255/255,111/255,105/255,1),(160/255,64/255,160/255,1),(205/255,62/255,205/255,1),(46/255,117/255,182/255,1),(52/255,152/255,219/255,1)]
    reg = ['0°N~30°N','30°N~60°N','60°N~90°N']
    for ran in range(3):
        # plt.plot(np.arange(1975,2016,1),aveUp[:, ran],colors[ran],label = str(ran * 30) + '-' + str(ran * 30 + 30) + 'increase')
        # plt.plot(np.arange(1975,2016,1),aveDown[:, ran],colors[ran] + '--', label = str(ran * 30) + '-' + str(ran * 30 + 30) + 'decrease')
        
        ax_three.bar(np.arange(1975,2026-12,1) + 0.3 * ran, aveUp[:, ran], color = colors[ran * 2],width = 0.3, label = 'Shifting into {}'.format(reg[ran]))
        ax_three.bar(np.arange(1975,2026-12,1) + 0.3 * ran, -aveDown[:, ran], color = colors[ran * 2 + 1], width = 0.3, label = 'Shifting out of {}'.format(reg[ran]))
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
    font3 = {
         'weight': 'bold',
		 'style':'normal',
         'size': 10,
         }
    font4 = {
         'weight': 'bold',
		 'style':'normal',
         'size': 30,
         }
    font5 = {
		 'style':'normal',
         'size': 10,
         }
    font6 = {
         'weight': 'bold',
		 'style':'normal',
         'size': 20,
         }
    # plt.ylabel('Average family',fontsize = 20)
    ax_three.set_ylabel('Percent Species Shifting',font1,labelpad=15)
    ax_three.set_xticks([1975,1984,1993,2002,2011])
    ax_three.set_yticks([-0.04,-0.02,0,0.02,0.04])

    ax_three.set_xticklabels(['1970-1982','1979-1991','1988-2000','1997-2009','2006-2018'],font1)
    ax_three.set_yticklabels(['4%','2%','0%','2%','4%'],font1)
    ax_three.set_xlabel('Year',font1)
    ax_three.set_xlim([1972,2017])
    ax_three.set_ylim([-0.055,0.055])
    ax_three.legend(prop = font2,ncol = 3,loc = 3,fontsize = 12, columnspacing = 0.5)
    set_axis(ax_three)
    ax_three.text(1973, -0.055 + (0.11) * 0.96, 'C', verticalalignment="top",
                horizontalalignment="left", fontdict=font4)
    ax_three.text(1975, -0.055 + (0.11) * 0.95, 'At the taxonomic level of genus', verticalalignment="top",
                horizontalalignment="left", fontdict=font6)
    # fig = plt.gcf()
    # fig.set_size_inches(20, 10)

    '''
    LOWESS regression
    '''
    def get_p_value(arrA, arrB):
        a = np.array(arrA)
        b = np.array(arrB)
        t, p = stats.ttest_ind(a,b)
        return 1 - p

    left,bottom,width,height = [0.68,0.65,0.3,0.3]
    # ax2 = fig.add_axes([left,bottom,width,height])
    ax2 = ax_three.inset_axes((left,bottom,width,height))

    left,bottom,width,height = [0.68,0.05,0.3,0.3]
    # ax3 = fig.add_axes([left,bottom,width,height])
    ax3 = ax_three.inset_axes((left,bottom,width,height))
    for ran in range(3):
        for half in range(2):
            smoother = LowessSmoother(smooth_fraction=0.2, iterations=1)
            if half == 0:
                initial = aveUp[:, ran]     
            else:
                initial = aveDown[:, ran]
            smoother.smooth(initial)
            y_pred = smoother.smooth_data[0]
            if half == 0:
                ax2.plot(np.arange(1975,2026-12,1), y_pred, color = colors[ran * 2], label ='$\\it{p}$ = ' + '%.4f' % get_p_value(initial, y_pred))
            else:
                ax3.plot(np.arange(1975,2026-12,1), y_pred, '--', color = colors[ran * 2 + 1], label = '$\\it{p}$ = ' +'%.4f' % get_p_value(initial, y_pred))
    ax2.legend(prop = font5,loc=1, fontsize = 10,labelspacing = 0.2)
    ax3.legend(prop = font5,loc=1, fontsize = 10,labelspacing = 0.2)

    ax2.set_xticks([1980,1995,2010])
    ax2.set_yticks([0,0.015,0.03])
    ax2.set_xticklabels(['1975-1987','1990-2002','2005-2017'],font3)
    ax2.set_yticklabels(['0%','1.5%','3%'],font3)
    ax2.set_ylim([-0.005,0.035])
    ax2.grid(False)
    ax3.set_xticks([1980,1995,2010])
    ax3.set_yticks([0,0.015,0.03])
    ax3.set_xticklabels(['1975-1987','1990-2002','2005-2017'],font3)
    ax3.set_yticklabels(['0%','1.5%','3%'],font3)
    ax3.set_ylim([-0.005,0.035])
    ax3.grid(False)

    ax2.spines['top'].set_color('black')
    ax2.spines['bottom'].set_color('black')
    ax2.spines['left'].set_color('black')
    ax2.spines['right'].set_color('black')
    ax3.spines['top'].set_color('black')
    ax3.spines['bottom'].set_color('black')
    ax3.spines['left'].set_color('black')
    ax3.spines['right'].set_color('black')

    ax2.text(1975,-0.003,'C$_1$ Fitting trends (Shifting into regions)',verticalalignment="bottom",horizontalalignment="left",fontdict=font2)
    ax3.text(1975,-0.003,'C$_2$ Fitting trends (Shifting out of regions)',verticalalignment="bottom",horizontalalignment="left",fontdict=font2)
    for tickline in ax2.xaxis.get_ticklines():
        tickline.set_visible(True)
    for tickline in ax2.yaxis.get_ticklines():
        tickline.set_visible(True)
    for tickline in ax3.xaxis.get_ticklines():
        tickline.set_visible(True)
    for tickline in ax3.yaxis.get_ticklines():
        tickline.set_visible(True)
    # fig.savefig(r'chugao\fish_220114_re\correlation_median_point\three_1.png',dpi=1000)
    # plt.show()
    # for year in range(yearTotal - 10):
    #     for ran in range(3):
    #         aveUp[year, ran] = np.mean(aveFamilyUp[year: year + 10, ran][aveFamilyUp[year: year + 10, ran] > 0])
    #         aveDown[year, ran] = np.mean(aveFamilyDown[year: year + 10, ran][aveFamilyDown[year: year + 10, ran] > 0])
    # for ran in range(3):
    #     plt.plot(np.arange(1975,2016,1),aveUp[:, ran],colors[ran],label = str(ran * 30) + '-' + str(ran * 30 + 30) + 'increase')
    #     plt.plot(np.arange(1975,2016,1),aveDown[:, ran],colors[ran] + '--', label = str(ran * 30) + '-' + str(ran * 30 + 30) + 'decrease')

    # plt.ylabel('Average family',fontsize = 25)
    # plt.xticks(fontsize = 25)
    # plt.yticks(fontsize = 25)
    # plt.xlabel('Year',fontsize = 25)
    # plt.legend(ncol = 3,loc = 4,fontsize = 25)
    # plt.grid()
    # fig = plt.gcf()
    # fig.set_size_inches(20, 10)
    # fig.savefig(r'chugao\fish_220114_re\correlation_median_point\four_1.png',dpi=1000)
    # plt.show()
