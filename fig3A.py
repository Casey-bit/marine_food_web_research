from matplotlib import markers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from collections import Counter  # 引入Counter
import pickle
from functools import reduce
import seaborn as sns
import scipy
import operator
from mpl_toolkits.mplot3d import Axes3D
import tot_visualization
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

sns.set_style("whitegrid")


def save_variable(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()
    return filename


def load_variavle(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
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
    for tickline in ax.yaxis.get_minorticklines():
        tickline.set_visible(True)
    ax.tick_params(which="major",
                   length=15, width=2.0,
                   colors="black",
                   direction='in',
                   tick2On=False,
                   label2On=False)
    ax.tick_params(which="minor",
                   length=5, width=1.0,
                   colors="black",
                   direction='in',
                   tick2On=False,
                   label2On=False)


def one_two(ax_one, ax_subone, ax_two):

    han, lab = tot_visualization.sub_one(ax_subone)
    # dffamily = pd.read_csv('E:\\paper\\obis_20220114.csv\\GenusNum_1.csv')

    dffamily = pd.read_csv('E:\\paper\\obis_20220114.csv\\FamilyNum_1.csv')

    family = dffamily['family'].values.tolist()

    start = 1970
    end = 2020
    yearTotal = end - start + 1
    familyNum = 2846

    # year family range_1_degree
    countByYear = load_variavle(
        'chugao\\fish_220114_re\\countByYear_1degree_1970_2020_{}_1.txt'.format(familyNum))
    fifty = load_variavle(
        r'chugao\fish_220114_re\correlation_median_point\medianPoint.txt')
    # meanFifty = np.array([[0.0 for year in range(yearTotal - 4)] for r in range(3)]) # 存放三个区域的50%平均轨迹
    # countFifty = np.array([[0 for year in range(yearTotal - 4)] for r in range(3)]) # 存放三个区域的50%物种数量
    meanLatitude = [[[] for year in range(yearTotal - 2)]
                    for r in range(6)]  # 存放三个区域的物种平均纬度
    countLatitude = np.array([[0 for year in range(yearTotal - 2)]
                             for r in range(6)])  # 存放三个区域的物种数量
    # for year in range(yearTotal - 4):
    #     for family in range(familyNum):
    #         latitude = fifty[family, year]
    #         if latitude <= 30:
    #             meanFifty[0, year] += np.mean()
    #             countFifty[0, year] += 1
    #         elif 30 < latitude <= 60:
    #             meanFifty[1, year] += latitude
    #             countFifty[1, year] += 1
    #         elif 60 < latitude <= 90:
    #             meanFifty[2, year] += latitude
    #             countFifty[2, year] += 1

    # meanFifty = meanFifty / countFifty

    # for i in range(6):
    #     plt.plot(np.arange(1970, 2021 - 4, 1), meanFifty[i])
    # plt.show()

    reserve = load_variavle(
        r'chugao\fish_220114_re\correlation_median_point\reserve.txt')

    for year in range(yearTotal - 2):
        for family in range(familyNum):
            l = [i for i, j in reserve if j == family]
            if len(l) > 0:
                if np.mean(fifty[family, year: year + 2][fifty[family, year: year + 2] > 0]) < np.mean(fifty[family, year + 1: year + 3][fifty[family, year + 1: year + 3] > 0]):
                    # 上移
                    move = 0
                elif np.mean(fifty[family, year: year + 2][fifty[family, year: year + 2] > 0]) > np.mean(fifty[family, year + 1: year + 3][fifty[family, year + 1: year + 3] > 0]):
                    # 下移
                    move = 1
                else:
                    move = -1  # 均值为Nan时
                ranges = np.array([np.sum(countByYear[year: year + 3, family, 90:120]), np.sum(
                    countByYear[year: year + 3, family, 120:150]), np.sum(countByYear[year: year + 3, family, 150:])])
                if ranges[0] == np.max(ranges):
                    c = 0
                elif ranges[1] == np.max(ranges):
                    c = 2
                elif ranges[2] == np.max(ranges):
                    c = 4
                sum = 0
                for r in range(90, 180):
                    sum += r * countByYear[year, family, r]
                if sum > 0 and move > -1:
                    sum /= np.sum(countByYear[year, family, 90: 180])
                    meanLatitude[c + move][year].append(sum)

    meanLatitudeArray = np.array(
        [[0.0 for year in range(yearTotal - 2)] for r in range(6)])
    stdLatitudeArray = np.array(
        [[0.0 for year in range(yearTotal - 2)] for r in range(6)])
    for r in range(6):
        for year in range(yearTotal - 2):
            meanLatitudeArray[r, year] = np.mean(meanLatitude[r][year])
            stdLatitudeArray[r, year] = np.std(meanLatitude[r][year], ddof=1)
    ave = np.array([[0.0 for year in range(yearTotal - 11)] for r in range(6)])
    avestd = np.array([[0.0 for year in range(yearTotal - 11)]
                      for r in range(6)])
    for r in range(6):
        for year in range(yearTotal - 11):
            ave[r, year] = np.mean(meanLatitudeArray[r, year: year + 10]
                                   [~np.isnan(meanLatitudeArray[r, year: year + 10])])
            avestd[r, year] = np.mean(
                stdLatitudeArray[r, year: year + 10][~np.isnan(stdLatitudeArray[r, year: year + 10])])
    color2 = [(236/255, 95/255, 116/255, 0.8), (255/255, 111/255, 105/255, 0.8), (160/255, 64/255, 160/255, 0.8),
              (205/255, 62/255, 205/255, 0.8), (46/255, 117/255, 182/255, 0.8), (52/255, 152/255, 219/255, 0.8)]
    color3 = [(236/255, 95/255, 116/255, 0.2), (255/255, 111/255, 105/255, 0.2), (160/255, 64/255, 160/255, 0.2),
              (205/255, 62/255, 205/255, 0.2), (46/255, 117/255, 182/255, 0.2), (52/255, 152/255, 219/255, 0.2)]
    marker = ['^', 'v']
    style = ['-', '--']
    label = ['Family shift towards north', 'Family shift towards south', 'Family shift towards north',
             'Family shift towards south', 'Family shift towards north', 'Family shift towards south']
    s = [(500*(400-200)/(1000-200), 400), (500*(600-200) /
                                           (1000-200), 600), (500*(800-200)/(1000-200), 800)]
    for r in range(6):
        ax_one.plot([1, 2], [1, 2], style[r % 2], linewidth=3, color=color2[r],
                    label=label[r], marker=marker[r % 2], markersize=20, markeredgecolor=color2[r])
        ax_one.plot(np.arange(1975, 2026 - 11, 1),
                    ave[r] - 90, style[r % 2], color=color2[r], linewidth=3)
        ax_one.fill_between(np.arange(
            1975, 2026 - 11, 1), ave[r] - avestd[r] / 2 - 90, ave[r] + avestd[r] / 2 - 90, facecolor=color3[r])

        if r % 2 == 1:
            ax_one.scatter([], [], c='k', marker='^', alpha=0.8, s=s[int(r / 2)][0],
                           label='Triangle size is proportional to Family Index')
    meanFamilyup = np.array([[0.0 for year in range(yearTotal - 2)]
                            for r in range(3)])  # 存放三个区域的平均迁移物种
    meanFamilydown = np.array(
        [[0.0 for year in range(yearTotal - 2)] for r in range(3)])  # 存放三个区域的平均迁移物种

    familyupNum = np.array([[0.0 for year in range(yearTotal - 2)]
                           for r in range(3)])  # 存放三个区域的迁移物种的个数
    familydownNum = np.array([[0.0 for year in range(yearTotal - 2)]
                             for r in range(3)])  # 存放三个区域的迁移物种的个数

    familyUp = [[[] for r in range(3)] for year in range(yearTotal - 2)]
    familyDown = [[[] for r in range(3)] for year in range(yearTotal - 2)]

    for year in range(yearTotal - 2):
        for family in range(familyNum):
            l = [i for i, j in reserve if j == family]
            if len(l) > 0:
                latitude = fifty[family, year]
                ranges = np.array([np.sum(countByYear[year: year + 3, family, 90:120]), np.sum(
                    countByYear[year: year + 3, family, 120:150]), np.sum(countByYear[year: year + 3, family, 150:])])
                if ranges[0] == np.max(ranges):
                    if np.mean(fifty[family, year: year + 2][fifty[family, year: year + 2] > 0]) < np.mean(fifty[family, year + 1: year + 3][fifty[family, year + 1: year + 3] > 0]):
                        familyUp[year][0].append(l[0])
                    elif np.mean(fifty[family, year: year + 2][fifty[family, year: year + 2] > 0]) > np.mean(fifty[family, year + 1: year + 3][fifty[family, year + 1: year + 3] > 0]):
                        familyDown[year][0].append(l[0])
                if ranges[1] == np.max(ranges):
                    if np.mean(fifty[family, year: year + 2][fifty[family, year: year + 2] > 0]) < np.mean(fifty[family, year + 1: year + 3][fifty[family, year + 1: year + 3] > 0]):
                        familyUp[year][1].append(l[0])
                    elif np.mean(fifty[family, year: year + 2][fifty[family, year: year + 2] > 0]) > np.mean(fifty[family, year + 1: year + 3][fifty[family, year + 1: year + 3] > 0]):
                        familyDown[year][1].append(l[0])
                if ranges[2] == np.max(ranges):
                    if np.mean(fifty[family, year: year + 2][fifty[family, year: year + 2] > 0]) < np.mean(fifty[family, year + 1: year + 3][fifty[family, year + 1: year + 3] > 0]):
                        familyUp[year][2].append(l[0])
                    elif np.mean(fifty[family, year: year + 2][fifty[family, year: year + 2] > 0]) > np.mean(fifty[family, year + 1: year + 3][fifty[family, year + 1: year + 3] > 0]):
                        familyDown[year][2].append(l[0])

        for r in range(3):
            meanFamilyup[r, year] = np.mean(familyUp[year][r])
            meanFamilydown[r, year] = np.mean(familyDown[year][r])
            familyupNum[r, year] = len(familyUp[year][r])
            familydownNum[r, year] = len(familyDown[year][r])

    # save_variable(familyUp, r'chugao\fish_220114_re\correlation_median_point\localDelay\familyUp0.txt')
    # save_variable(familyDown, r'chugao\fish_220114_re\correlation_median_point\localDelay\familyDown0.txt')

    # meanFamilyup[meanFamilyup > 1500] = 1500
    # meanFamilydown[meanFamilydown > 1500] = 1500

    ave_arrow_up = np.array(
        [[0.0 for year in range(yearTotal - 11)] for r in range(6)])
    ave_arrow_down = np.array(
        [[0.0 for year in range(yearTotal - 11)] for r in range(6)])
    for r in range(3):
        for year in range(yearTotal - 11):
            ave_arrow_up[r, year] = np.mean(
                meanFamilyup[r, year: year + 10][~np.isnan(meanFamilyup[r, year: year + 10])])
            ave_arrow_down[r, year] = np.mean(
                meanFamilydown[r, year: year + 10][~np.isnan(meanFamilydown[r, year: year + 10])])

    arrowColor = [(236/255, 95/255, 116/255, 0.8), (255/255, 111/255, 105/255, 0.8), (160/255, 64/255, 160/255, 0.8),
                  (205/255, 62/255, 205/255, 0.8), (46/255, 117/255, 182/255, 0.8), (52/255, 152/255, 219/255, 0.8)]
    for r in range(3):
        for year in range(yearTotal - 11):
            if year % 3 == 0:
                ax_one.scatter(year + 1975, ave[2 * r][year] - 90, c=arrowColor[2 * r], marker='^', s=np.mean(
                    [500*(meanFamilyup[r, y] - 200) / (1000 - 200) for y in range(year, year + 4) if meanFamilyup[r, y] > 0]))
                ax_one.scatter(year + 1975, ave[2 * r + 1][year] - 90, c=arrowColor[2 * r + 1], marker='v', s=np.mean(
                    [500*(meanFamilydown[r, y] - 200) / (1000 - 200) for y in range(year, year + 4) if meanFamilydown[r, y] > 0]))
              # plt.arrow(year + 1975, ave[2 * r][year] - 90, 0, 0.1, width=meanFamilyup[r, year]/1500, color=arrowColor[2 * r],  length_includes_head=False)
            # plt.arrow(year + 1975, ave[2 * r + 1][year] - 90, 0, -0.1, width=meanFamilydown[r, year]/1500, color=arrowColor[2 * r + 1],  length_includes_head=False)
        # plt.plot(np.arange(1971, 2021 - 4, 1), meanFamily[i], label = str(i))
    # plt.legend()
    ax_one.set_ylim([0, 90])
    ax_one.set_xlim([1972, 2025])
    ax_one.set_xticks([1975, 1983, 1991, 1999, 2007, 2015])
    font1 = {
        'weight': 'bold',
        'style': 'normal',
        'size': 15,
    }
    font2 = {
        'weight': 'bold',
        'style': 'normal',
        'size': 12,
    }
    font3 = {
        'weight': 'bold',
        'style': 'normal',
        'size': 30,
    }
    ax_one.set_xticklabels(['1970-1982', '1978-1990', '1986-1998',
                           '1994-2006', '2002-2014', '2010-2022'], font1)
    # ax_one.set_xticklabels(ax_one.get_xticklabels())
    ax_one.set_yticks([0, 30, 60, 90])
    ax_one.set_yticks([10, 20, 40, 50, 70, 80], minor=True)

    ax_one.set_yticklabels(ax_one.get_yticks(), font1)
    ax_one.set_ylabel("Latitude (°N)", font1)
    # ax_one.yaxis.set_minor_locator(MultipleLocator(10))
    # ax_one.yaxis.set_yminorticklavbels()
    ax_one.set_xlabel("Year", font1)

    handles, labels = ax_one.get_legend_handles_labels()

    # handles = [handles[0], handles[1], handles[6],handles[2], handles[3], handles[7],handles[4], handles[5], handles[8]]
    # labels = [labels[0], labels[1], labels[6],labels[2], labels[3], labels[7],labels[4], labels[5], labels[8]]

    l1 = ax_one.legend([handles[0], handles[1], han[0], handles[2], handles[3], han[1], handles[4], handles[5], han[2]], [labels[0], labels[1], lab[0], labels[2], labels[3], lab[1], labels[4], labels[5], lab[2]],
                       prop=font2, ncol=3, loc=(0.07, 0.9), fontsize=12, markerscale=0.5, columnspacing=0.5)
    l2 = ax_one.legend(handles[6:7], labels[6:7],
                       prop=font2, ncol=3, loc=(0.07, 0.85), fontsize=12, columnspacing=0.5)
    ax_one.add_artist(l1)
    ax_one.annotate(s='', xy=(1973, 2), xytext=(1973, 28), arrowprops=dict(
        color='black', arrowstyle='<->,head_width=0.5,head_length = 1.5', linewidth=3))
    ax_one.annotate(s='', xy=(1973, 32), xytext=(1973, 58), arrowprops=dict(
        color='black', arrowstyle='<->,head_width=0.5,head_length = 1.5', linewidth=3))
    ax_one.annotate(s='', xy=(1973, 62), xytext=(1973, 88), arrowprops=dict(
        color='black', arrowstyle='<->,head_width=0.5,head_length = 1.5', linewidth=3))
    ax_one.text(1973.7, 11, 'Region 1', rotation=90,
                fontsize=12, weight='bold')
    ax_one.text(1973.7, 41, 'Region 2', rotation=90,
                fontsize=12, weight='bold')
    ax_one.text(1973.7, 71, 'Region 3', rotation=90,
                fontsize=12, weight='bold')
    set_axis(ax_one)
    # ax_one.text(1974, 88, 'A', verticalalignment="top",
    #             horizontalalignment="left", fontdict=font3)

    # fig = plt.gcf()
    # fig.set_size_inches(20, 10)
    # fig.savefig(r'chugao\fish_220114_re\correlation_median_point\one.png',dpi=1000)
    # plt.show()

    # a = [[] for r in range(3)]
    # b = [[] for r in range(3)]
    # for r in range(3):
    #     for year in range(yearTotal - 9):
    #         a[r].append(np.mean(meanFamilyup[r, year : year + 10][~np.isnan(meanFamilyup[r, year : year + 10])]))
    #         b[r].append(np.mean(meanFamilydown[r, year : year + 10][~np.isnan(meanFamilydown[r, year : year + 10])]))

    c = ['r', 'g', 'b']
    reg = ['0°N~30°N','30°N~60°N','60°N~90°N']
    for r in range(3):
        ax_two.plot(np.arange(1975, 2026-11, 1), ave_arrow_up[r], color=color2[2 * r],
                    linewidth=3, label='Family within {} (N)'.format(reg[r]))
        ax_two.plot(np.arange(1975, 2026-11, 1), ave_arrow_down[r], '--', color=color2[2 *
                    r + 1], linewidth=3, label='Family within {} (S)'.format(reg[r]))
        save_variable(ave_arrow_up[r],'chugao\\fish_220114_re\\figure2\\N_1_{}.txt'.format(r))
        save_variable(ave_arrow_down[r],'chugao\\fish_220114_re\\figure2\\S_1_{}.txt'.format(r))
    ax_two.set_xticks([1980, 1995, 2010])
    ax_two.set_xticklabels(['1975-1987', '1990-2002', '2005-2017'], font1)
    ax_two.set_yticks([200, 400, 600, 800, 1000])
    # ax_two.set_xticklabels(ax_two.get_xticklabels())

    ax_two.set_yticklabels(ax_two.get_yticks(), font1)
    ax_two.set_ylabel("Average Family Index", font1)
    ax_two.set_xlabel("Year", font1)
    ax_two.set_xlim([1972, 2017])
    ax_two.set_ylim([100, 1000])
    ax_two.legend(prop=font2, ncol=1, loc=4, fontsize=12, columnspacing=0.5)
    # ax_two.set_title('Within Regions',font1)
    ax_two.text(1978, 970, 'Within Regions', verticalalignment="top",
                horizontalalignment="left", fontdict=font1)
    ax_two.text(1973, 970, 'A', verticalalignment="top",
                horizontalalignment="left", fontdict=font3)
    set_axis(ax_two)

    # fig = plt.gcf()
    # fig.set_size_inches(20, 10)
    # fig.savefig(r'chugao\fish_220114_re\correlation_median_point\two.png',dpi=1000)
    # plt.show()

    # a = np.array([[0 for year in range(yearTotal - 5)] for r in range(6)])
    # a = [[] for r in range(3)]
    # b = [[] for r in range(3)]
    # c = [[] for r in range(3)]
    # for r in range(3):
    #     for year in range(yearTotal - 14):
    #         a[r].append(np.mean(familyupNum[r, year : year + 10][familyupNum[r, year : year + 10] > 0]))
    #         b[r].append(np.mean(familydownNum[r, year : year + 10][familydownNum[r, year : year + 10] > 0]))
    #         c[r].append(np.mean(familyupNum[r, year : year + 10][familyupNum[r, year : year + 10] > 0]) + np.mean(familydownNum[r, year : year + 10][familydownNum[r, year : year + 10] > 0]))
    #         # a[r, year] = np.mean(familyupNum[r, year : year + 10]) - np.mean(familydownNum[r, year : year + 10])
    #         # a[r, year] = familyupNum[r, year] - familydownNum[r, year]
    # label = ['0-30up', '0-30down', '0-30total', '30-60dup', '30-60down', '30-60total', '60-90up', '60-90down', '60-90total']
    # for r in range(3):
    #     plt.bar(np.arange(1975,2025-13), a[r], width = 0.3, alpha = 0.75, label = label[3 * r])
    #     plt.bar(np.arange(1975,2025-13) + 0.3, b[r], width = 0.3, alpha = 0.75, label = label[3 * r + 1])
    #     plt.bar(np.arange(1975,2025-13) + 0.6, c[r], width = 0.3, alpha = 0.75, label = label[3 * r + 2])
    #     plt.legend(fontsize = 20)
    #     plt.xticks(fontsize = 20)
    #     plt.yticks(fontsize = 20)
    #     plt.show()

    # plt.show()
