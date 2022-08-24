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
from statsmodels.formula.api import ols
import pickle


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
    ax.tick_params(which="major",
                   length=15, width=2.0,
                   colors="black",
                   direction='in',
                   tick2On=False)


def level(ax, index):

    if index == 3:
        up = load_variavle(
            r'chugao\fish_220114_re\figure2\extra\family_into_familyindex.txt')
        down = load_variavle(
            r'chugao\fish_220114_re\figure2\extra\family_out_of_familyindex.txt')
    else:
        up = load_variavle(
            r'chugao\fish_220114_re\figure2\extra\family_into_familylevel.txt')
        down = load_variavle(
            r'chugao\fish_220114_re\figure2\extra\family_out_of_familylevel.txt')

    slide = 15

    up_slide = [[[] for r in range(3)] for year in range(50 - slide)]
    down_slide = [[[] for r in range(3)] for year in range(50 - slide)]

    for y in range(50 - slide):
        for r in range(3):
            for idx in range(y, y + slide):
                up_slide[y][r] += up[idx][r]
                down_slide[y][r] += down[idx][r]

    up_slide_value = [[0.0 for r in range(3)] for year in range(50 - slide)]
    down_slide_value = [[0.0 for r in range(3)] for year in range(50 - slide)]

    for y in range(50 - slide):
        for r in range(3):
            if index == 2:
                up_slide_value[y][r] = len([i for i in up_slide[y][r] if i > 2]) / (
                    len([i for i in up_slide[y][r] if i <= 2]) + 1e-6)
                down_slide_value[y][r] = len([i for i in down_slide[y][r] if i > 2]) / (
                    len([i for i in down_slide[y][r] if i <= 2]) + 1e-6)
            else:
                up_slide_value[y][r] = np.mean(up_slide[y][r])
                down_slide_value[y][r] = np.mean(down_slide[y][r])

    colors = [(236/255, 95/255, 116/255, 0.8), (255/255, 111/255, 105/255, 0.8),
              (160/255, 64/255, 160/255, 0.8), (205/255, 62/255, 205/255, 0.8),
              (46/255, 117/255, 182/255, 0.8), (52/255, 152/255, 219/255, 0.8)]
    reg = ['0°N~30°N', '30°N~60°N', '60°N~90°N']

    for r in range(3):
        df = pd.DataFrame({'x': np.arange(1977, 1977 + 50 - slide),
                          'y': [up_slide_value[y][r] for y in range(50 - slide)]})
        model = ols("y ~ x", df).fit()
        k = model.params[1]  # 拟合直线斜率
        b = model.params[0]  # 截距
        y_pred = b + k * np.arange(1977, 1977 + 50 - slide)  # 预测值
        k = "%.2e" % k
        b = "%.2e" % b
        line = 'y=(' + k + ')x+(' + b + ')'
        R2 = model.rsquared
        P = model.pvalues[1]
        print(P)
        ax.plot(np.arange(1977, 1977 + 50 - slide), y_pred, color=colors[2 * r],
                linewidth=3, label='Shifting into {}'.format(reg[r]))

        df = pd.DataFrame({'x': np.arange(1977, 1977 + 50 - slide),
                          'y': [down_slide_value[y][r] for y in range(50 - slide)]})
        model = ols("y ~ x", df).fit()
        k = model.params[1]  # 拟合直线斜率
        b = model.params[0]  # 截距
        y_pred = b + k * np.arange(1977, 1977 + 50 - slide)  # 预测值
        k = "%.2e" % k
        b = "%.2e" % b
        line = 'y=(' + k + ')x+(' + b + ')'
        R2 = model.rsquared
        P = model.pvalues[1]
        print(P)
        ax.plot(np.arange(1977, 1977 + 50 - slide), y_pred, '--', color=colors[2 * r],
                linewidth=3, label='Shifting out of {}'.format(reg[r]))

        ax.plot(np.arange(1977, 1977 + 50 - slide), [up_slide_value[y][r] for y in range(
            50 - slide)], color=colors[2 * r], linewidth=2, alpha=0.5)
        ax.plot(np.arange(1977, 1977 + 50 - slide), [down_slide_value[y][r] for y in range(
            50 - slide)], '--', color=colors[2 * r + 1], linewidth=2, alpha=0.5)
        # ax.plot(np.arange(1977, 1977 + 50 - slide), [up_slide_value[y][r] for y in range(50 - slide)], color = colors[2 * r], linewidth = 3, label = 'Shifting into {}'.format(reg[r]))
        # ax.plot(np.arange(1977, 1977 + 50 - slide), [down_slide_value[y][r] for y in range(50 - slide)], '--', color = colors[2 * r + 1], linewidth = 3, label = 'Shifting out of {}'.format(reg[r]))

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
    ax.set_xticks([1980, 1995, 2010])
    ax.set_xticklabels(['1975-1990', '1990-2005', '2005-2020'], font1)

    if index == 3:
        ax.set_ylabel('Average Family Index', font1)
        ax.set_ylim([350, 850])
        ax.set_yticks([400, 500, 600, 700, 800])
        ax.set_yticklabels(ax.get_yticks(), font1)
        ax.text(1974, 350 + (850 - 350) * 0.97, 'C', verticalalignment="top",
                horizontalalignment="left", fontdict=font3)
        ax.text(1978.5, 350 + (850 - 350) * 0.96, r'Linear fitting $(p\rm<0.001)$', verticalalignment="top",
                horizontalalignment="left", fontdict=font1)
    elif index == 1:
        ax.set_ylabel('Average Family Trophic Level', font1)
        ax.set_ylim([2.1, 2.35])
        ax.set_yticks([2.15, 2.2, 2.25, 2.3])
        ax.set_yticklabels(ax.get_yticks(), font1)
        ax.text(1974, 2.1 + (2.35 - 2.1) * 0.97, 'A', verticalalignment="top",
                horizontalalignment="left", fontdict=font3)
        ax.text(1978.5, 2.1 + (2.35 - 2.1) * 0.96, r'Linear fitting $(p\rm<0.001)$', verticalalignment="top",
                horizontalalignment="left", fontdict=font1)
    elif index == 2:
        ax.set_ylabel(
            'Ratio of Species at\nHigher Trophic Levels to Lower Levels', font1)
        ax.set_ylim([0.2, 0.55])
        ax.set_yticks([0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
        ax.set_yticklabels(ax.get_yticks(), font1)
        ax.text(1974, 0.2 + (0.55 - 0.2) * 0.97, 'B', verticalalignment="top",
                horizontalalignment="left", fontdict=font3)
        ax.text(1978.5, 0.2 + (0.55 - 0.2) * 0.96, r'Linear fitting $(p\rm<0.001)$', verticalalignment="top",
                horizontalalignment="left", fontdict=font1)

    ax.legend(prop=font2, ncol=1, loc=4, fontsize=12, columnspacing=0.5)
    ax.set_xlabel('Year', font1)
    ax.set_xlim([1972, 2015])

    set_axis(ax)
