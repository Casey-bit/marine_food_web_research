'''
 # Copyright (C) 2022-08-09 School of IoT, Jiangnan University Limited
 #
 # All Rights Reserved.
 #
 # Paper: Successive Relocation of Marine Species in the Northern Hemisphere from 1970 to 2020
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

def four(ax_four):

    start = 1970
    end = 2020
    yearTotal = end - start + 1

    # year family range
    percent = load_variavle(r'chugao\fish_220114_re\cluster\percent_order_657.txt')
    reserve = load_variavle(r'chugao\fish_220114_re\correlation_median_point\reserve_order.txt')
    countByYear = load_variavle('chugao\\fish_220114_re\\countByYear_1degree_order_1970_2020_657_1.txt')
    record = np.array(countByYear)

    percentUp = [[[] for ran in range(3)] for year in range(yearTotal - 1)]
    percentDown = [[[] for ran in range(3)] for year in range(yearTotal - 1)] # 记录三个区间每年比例上升下降的物种
    familyUp = [[[] for ran in range(3)] for year in range(yearTotal - 1)]
    familyDown = [[[] for ran in range(3)] for year in range(yearTotal - 1)]

    for year in range(yearTotal - 1):
        total = np.sum([record[year: year + 2, old, 90:] for new, old in reserve])
        print(year,total)
        for family in range(657):
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

    # save_variable(familyUp,r'chugao\extra\genusUp.txt')
    # save_variable(familyDown,r'chugao\extra\genusDown.txt')

    # aveFamilyUp = load_variavle(r'chugao\fish_220114_re\cluster\aveFamilyUpfour.txt')
    # aveFamilyDown = load_variavle(r'chugao\fish_220114_re\cluster\aveFamilyDownfour.txt')
    # aveFamilyUpNum = load_variavle(r'chugao\fish_220114_re\cluster\aveFamilyUpNum.txt')
    # aveFamilyDownNum = load_variavle(r'chugao\fish_220114_re\cluster\aveFamilyDownNum.txt')

    # 滑动
    aveUp = np.array([[0.0 for ran in range(3)] for year in range(yearTotal - 10)])
    aveDown = np.array([[0.0 for ran in range(3)] for year in range(yearTotal - 10)])
    # for year in range(yearTotal - 10):
    #     for ran in range(3):
    #         # aveUp[year, ran] = np.mean(aveFamilyUp[year: year + 10, ran][aveFamilyUp[year: year + 10, ran] > 0])
    #         # aveDown[year, ran] = np.mean(aveFamilyDown[year: year + 10, ran][aveFamilyDown[year: year + 10, ran] > 0])

    #         aveUp[year, ran] = np.mean(aveFamilyUpNum[year: year + 10, ran][aveFamilyUpNum[year: year + 10, ran] > 0])
    #         aveDown[year, ran] = np.mean(aveFamilyDownNum[year: year + 10, ran][aveFamilyUpNum[year: year + 10, ran] > 0])


    colors = [(236/255,95/255,116/255,0.8),(255/255,111/255,105/255,0.8),(160/255,64/255,160/255,0.8),(205/255,62/255,205/255,0.8),(46/255,117/255,182/255,0.8),(52/255,152/255,219/255,0.8)]
    # fig,ax1 = plt.subplots()

    # for ran in range(3):
    #     # plt.plot(np.arange(1975,2016,1),aveUp[:, ran],colors[ran],label = str(ran * 30) + '-' + str(ran * 30 + 30) + 'increase')
    #     # plt.plot(np.arange(1975,2016,1),aveDown[:, ran],colors[ran] + '--', label = str(ran * 30) + '-' + str(ran * 30 + 30) + 'decrease')
        
    #     ax1.bar(np.arange(1975,2026-10,1) + 0.3 * ran, aveUp[:, ran], color = colors[ran * 2],width = 0.3, alpha = 0.75, label = str(ran * 30) + '-' + str(ran * 30 + 30) + 'increase')
    #     ax1.bar(np.arange(1975,2026-10,1) + 0.3 * ran, -aveDown[:, ran], color = colors[ran * 2 + 1], width = 0.3, alpha = 0.75, label = str(ran * 30) + '-' + str(ran * 30 + 30) + 'decrease')

    # # plt.ylabel('Average family',fontsize = 20)
    # ax1.set_ylabel('Quantity',fontsize = 25)
    # ax1.set_xticks([1975,1983,1991,1999,2007,2015])
    # ax1.set_yticks([-0.03,-0.02,-0.01,0,0.01,0.02,0.03])
    # ax1.set_xticklabels(['1970-1982','1978-1990','1986-1998','1994-2006','2002-2014','2010-2022'],fontsize = 25)
    # ax1.set_yticklabels(['3%','2%','1%','0%','1%','2%','3%'],fontsize = 25)
    # ax1.set_xlabel('Year',fontsize = 25)
    # ax1.legend(ncol = 3,loc = 3,fontsize = 15)
    # ax1.grid()
    # fig = plt.gcf()
    # fig.set_size_inches(20, 10)

    # '''
    # LOWESS regression
    # '''
    # def get_p_value(arrA, arrB):
    #     a = np.array(arrA)
    #     b = np.array(arrB)
    #     t, p = stats.ttest_ind(a,b)
    #     return 1 - p

    # left,bottom,width,height = [0.62,0.62,0.25,0.24]
    # ax2 = fig.add_axes([left,bottom,width,height])

    # left,bottom,width,height = [0.62,0.18,0.25,0.24]
    # ax3 = fig.add_axes([left,bottom,width,height])

    # for ran in range(3):
    #     for half in range(2):
    #         smoother = LowessSmoother(smooth_fraction=0.8, iterations=1)
    #         if half == 0:
    #             initial = aveUp[:, ran]     
    #         else:
    #             initial = aveDown[:, ran]
    #         smoother.smooth(initial)
    #         y_pred = smoother.smooth_data[0]
    #         if half == 0:
    #             ax2.plot(np.arange(1975,2026-10,1), y_pred, color = colors[ran * 2], label ='P = ' + '%.4f' % get_p_value(initial, y_pred))
    #         else:
    #             ax3.plot(np.arange(1975,2026-10,1), -y_pred, color = colors[ran * 2 + 1], label = 'P = ' +'%.4f' % get_p_value(initial, y_pred))
    # ax2.legend(loc=1, fontsize = 15)
    # ax3.legend(loc=4, fontsize = 15)

    # ax2.set_xticks([1975,1987,1999,2011])
    # ax2.set_yticks([0,0.01,0.02])
    # ax2.set_xticklabels(['1970-1982','1982-1994','1994-2006','2006-2018'],fontsize = 15)
    # ax2.set_yticklabels(['0%','1%','2%'],fontsize = 15)
    # ax2.set_ylim([-0.005,0.025])

    # ax3.set_xticks([1975,1987,1999,2011])
    # ax3.set_yticks([-0.02,-0.01,0])
    # ax3.set_xticklabels(['1970-1982','1982-1994','1994-2006','2006-2018'],fontsize = 15)
    # ax3.set_yticklabels(['2%','1%','0%'],fontsize = 15)
    # ax3.set_ylim([-0.025,0.005])
    # # fig.savefig(r'chugao\fish_220114_re\correlation_median_point\three_1.png',dpi=1000)
    # plt.show()
    reg = ['0°N~30°N','30°N~60°N','60°N~90°N']
    for year in range(yearTotal - 10):
        for ran in range(3):
            aveUp[year, ran] = np.mean(aveFamilyUp[year: year + 10, ran][aveFamilyUp[year: year + 10, ran] > 0])
            aveDown[year, ran] = np.mean(aveFamilyDown[year: year + 10, ran][aveFamilyDown[year: year + 10, ran] > 0])
    for ran in range(3):
        ax_four.plot(np.arange(1975,2026-10,1),aveUp[:, ran],color = colors[2 * ran],linewidth = 3,label = 'Shifting into {}'.format(reg[ran]))
        ax_four.plot(np.arange(1975,2026-10,1),aveDown[:, ran],'--',linewidth = 3, color = colors[2 * ran + 1], label = 'Shifting out of {}'.format(reg[ran]))
        # save_variable(aveUp[:, ran],'chugao\\fish_220114_re\\figure2\\N_2_{}.txt'.format(ran))
        # save_variable(aveDown[:, ran],'chugao\\fish_220114_re\\figure2\\S_2_{}.txt'.format(ran))
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
         'size': 30,
         }
    font4 = {
         'weight': 'bold',
		 'style':'normal',
         'size': 20,
         }
    ax_four.set_xticks([1980, 1995, 2010])
    ax_four.set_xticklabels(['1975-1990', '1990-2005', '2005-2020'], font1)
    ax_four.set_yticks([100, 150, 200, 250, 300])
    ax_four.set_ylabel('Average Order Index',font1)
    ax_four.set_yticklabels(ax_four.get_yticks(), font1)
    ax_four.set_xlabel('Year',font1)
    ax_four.set_xlim([1972,2017])
    ax_four.set_ylim([50, 300])
    ax_four.legend(prop = font2,ncol = 1,loc = 4,fontsize = 12, columnspacing = 0.5)
    # ax_four.set_title('Across Regions',font1)
    # ax_four.text(1978, 290,'Across Regions',verticalalignment="top",horizontalalignment="left",fontdict=font1)
    ax_four.text(1974, 50 + (300 - 50) * 0.96, 'B', verticalalignment="top",
                horizontalalignment="left", fontdict=font3)
    ax_four.text(1978.5, 50 + (300 - 50) * 0.95, 'At the taxonomic level of order', verticalalignment="top",
                horizontalalignment="left", fontdict=font4)
    set_axis(ax_four)
    # fig = plt.gcf()
    # fig.set_size_inches(20, 10)
    # fig.savefig(r'chugao\fish_220114_re\correlation_median_point\four_1.png',dpi=1000)
    # plt.show()
