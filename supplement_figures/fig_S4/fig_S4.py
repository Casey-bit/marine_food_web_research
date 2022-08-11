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
import pickle
import seaborn as sns
import fig_S4_sub

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


def one_two(ax_one, ax_subone):

    han, lab = fig_S4_sub.sub_one(ax_subone)
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
    meanLatitude = [[[] for year in range(yearTotal - 2)]
                    for r in range(6)] 

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
                    move = -1  # if Nan
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
                            for r in range(3)]) 
    meanFamilydown = np.array(
        [[0.0 for year in range(yearTotal - 2)] for r in range(3)])  

    familyupNum = np.array([[0.0 for year in range(yearTotal - 2)]
                           for r in range(3)]) 
    familydownNum = np.array([[0.0 for year in range(yearTotal - 2)]
                             for r in range(3)])  

    familyUp = [[[] for r in range(3)] for year in range(yearTotal - 2)]
    familyDown = [[[] for r in range(3)] for year in range(yearTotal - 2)]

    reserve1 = load_variavle(r'chugao\fish_220114_re\figure2\extra\reserve.txt')

    for year in range(yearTotal - 2):
        for family in range(familyNum):
            l = [i for i, j in reserve1 if j == family]
            if len(l) > 0:
                latitude = fifty[family, year]
                ranges = np.array([np.sum(countByYear[year: year + 3, family, 90:120]), np.sum(
                    countByYear[year: year + 3, family, 120:150]), np.sum(countByYear[year: year + 3, family, 150:])])
                if ranges[0] == np.max(ranges):
                    if np.mean(fifty[family, year: year + 2][fifty[family, year: year + 2] > 0]) < np.mean(fifty[family, year + 1: year + 3][fifty[family, year + 1: year + 3] > 0]):
                        familyUp[year][0].append(l[0])
                    elif np.mean(fifty[family, year: year + 2][fifty[family, year: year + 2] > 0])> np.mean(fifty[family, year + 1: year + 3][fifty[family, year + 1: year + 3] > 0]):
                        familyDown[year][0].append(l[0])
                if ranges[1] == np.max(ranges):
                    if np.mean(fifty[family, year: year + 2][fifty[family, year: year + 2] > 0]) < np.mean(fifty[family, year + 1: year + 3][fifty[family, year + 1: year + 3] > 0]):
                        familyUp[year][1].append(l[0])
                    elif np.mean(fifty[family, year: year + 2][fifty[family, year: year + 2] > 0])> np.mean(fifty[family, year + 1: year + 3][fifty[family, year + 1: year + 3] > 0]):
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

    # save_variable(familyUp, r'chugao\fish_220114_re\figure2\extra\familyUpA0804.txt')
    # save_variable(familyDown, r'chugao\fish_220114_re\figure2\extra\familyDownA0804.txt')

    # save_variable(familyUp, r'chugao\extra\familyUpA.txt')
    # save_variable(familyDown, r'chugao\extra\familyDownA.txt')

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
    print('==================')
    print(lab)
    print('==================')
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
