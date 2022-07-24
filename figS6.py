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

reg = ['0°N~30°N','30°N~60°N','60°N~90°N']
color2 = [(236/255, 95/255, 116/255, 0.8), (255/255, 111/255, 105/255, 0.8), (160/255, 64/255, 160/255, 0.8),
            (205/255, 62/255, 205/255, 0.8), (46/255, 117/255, 182/255, 0.8), (52/255, 152/255, 219/255, 0.8)]
ax = plt.subplot(121)
for i in range(3):
    ave_arrow_up = load_variavle('chugao\\fish_220114_re\\figure2\\N_1_{}.txt'.format(i))
    ave_arrow_down = load_variavle('chugao\\fish_220114_re\\figure2\\S_1_{}.txt'.format(i))
    df = pd.DataFrame({'x':np.arange(1975, 2026-11, 1),'y':ave_arrow_up})
    model = ols("y ~ x",df).fit()
    k = model.params[1] #拟合直线斜率
    b = model.params[0] #截距
    y_pred = b + k * np.arange(1975, 2026-11, 1) #预测值
    k = "%.2e" % k
    b = "%.2e" % b
    line = 'y=(' + k + ')x+(' + b + ')'
    R2 = model.rsquared
    P = model.pvalues[1]
    lineRegression, = ax.plot(np.arange(1975, 2026-11, 1), y_pred, color=color2[2 * i],
                    linewidth=3, label='Family within {} (N)'.format(reg[i]))
    # plt.legend(handles=[lineRegression], loc=2, fontsize = 10, framealpha = 0.5)

    df = pd.DataFrame({'x':np.arange(1975, 2026-11, 1),'y':ave_arrow_down})
    model = ols("y ~ x",df).fit()
    k = model.params[1] #拟合直线斜率
    b = model.params[0] #截距
    y_pred = b + k * np.arange(1975, 2026-11, 1) #预测值
    k = "%.2e" % k
    b = "%.2e" % b
    line = 'y=(' + k + ')x+(' + b + ')'
    R2 = model.rsquared
    P = model.pvalues[1]
    lineRegression, = ax.plot(np.arange(1975, 2026-11, 1), y_pred, '--', color=color2[2 *i + 1],
                         linewidth=3, label='Family within {} (S)'.format(reg[i]))
    # plt.legend(handles=[lineRegression], loc=2, fontsize = 10, framealpha = 0.5)

    ax.plot(np.arange(1975, 2026-11, 1), ave_arrow_up, color=color2[2 * i],alpha = 0.5)
    ax.plot(np.arange(1975, 2026-11, 1), ave_arrow_down, '--',color = color2[2 *i + 1],alpha = 0.5)


ax.set_xticks([1980, 1995, 2010])
ax.set_xticklabels(['1975-1987', '1990-2002', '2005-2017'], font1)
ax.set_yticks([200, 400, 600, 800, 1000])
# ax_two.set_xticklabels(ax_two.get_xticklabels())

ax.set_yticklabels(ax.get_yticks(), font1)
ax.set_ylabel("Average Family Index", font1)
ax.set_xlabel("Year", font1)
ax.set_xlim([1972, 2017])
ax.set_ylim([100, 1000])
ax.legend(prop=font2, ncol=1, loc=4, fontsize=12, columnspacing=0.5)
# ax_two.set_title('Within Regions',font1)
ax.text(1978, 970, 'Within Regions (Linear fitting)', verticalalignment="top",
            horizontalalignment="left", fontdict=font1)
ax.text(1973, 970, 'A', verticalalignment="top",
            horizontalalignment="left", fontdict=font3)
set_axis(ax)


ax = plt.subplot(122)
for i in range(3):
    ave_arrow_up = load_variavle('chugao\\fish_220114_re\\figure2\\N_2_{}.txt'.format(i))
    ave_arrow_down = load_variavle('chugao\\fish_220114_re\\figure2\\S_2_{}.txt'.format(i))
    df = pd.DataFrame({'x':np.arange(1975,2026-10,1),'y':ave_arrow_up})
    model = ols("y ~ x",df).fit()
    k = model.params[1] #拟合直线斜率
    b = model.params[0] #截距
    y_pred = b + k * np.arange(1975,2026-10, 1) #预测值
    k = "%.2e" % k
    b = "%.2e" % b
    line = 'y=(' + k + ')x+(' + b + ')'
    R2 = model.rsquared
    P = model.pvalues[1]
    lineRegression, = ax.plot(np.arange(1975,2026-10, 1), y_pred, color=color2[2 * i],
                    linewidth=3, label='Family within {} (N)'.format(reg[i]))
    # plt.legend(handles=[lineRegression], loc=2, fontsize = 10, framealpha = 0.5)

    df = pd.DataFrame({'x':np.arange(1975,2026-10,1),'y':ave_arrow_down})
    model = ols("y ~ x",df).fit()
    k = model.params[1] #拟合直线斜率
    b = model.params[0] #截距
    y_pred = b + k * np.arange(1975,2026-10,1) #预测值
    k = "%.2e" % k
    b = "%.2e" % b
    line = 'y=(' + k + ')x+(' + b + ')'
    R2 = model.rsquared
    P = model.pvalues[1]
    lineRegression, = ax.plot(np.arange(1975,2026-10,1), y_pred, '--', color=color2[2 *i + 1],
                         linewidth=3, label='Family within {} (S)'.format(reg[i]))
    # plt.legend(handles=[lineRegression], loc=2, fontsize = 10, framealpha = 0.5)

    ax.plot(np.arange(1975,2026-10,1), ave_arrow_up, color=color2[2 * i],alpha = 0.5)
    ax.plot(np.arange(1975,2026-10,1), ave_arrow_down, '--',color = color2[2 *i + 1],alpha = 0.5)


ax.set_xticks([1980, 1995, 2010])
ax.set_xticklabels(['1975-1987', '1990-2002', '2005-2017'], font1)
ax.set_yticks([200, 400, 600, 800, 1000])
# ax_two.set_xticklabels(ax_two.get_xticklabels())

ax.set_yticklabels(ax.get_yticks(), font1)
ax.set_ylabel("Average Family Index", font1)
ax.set_xlabel("Year", font1)
ax.set_xlim([1972, 2017])
ax.set_ylim([100, 1000])
ax.legend(prop=font2, ncol=1, loc=4, fontsize=12, columnspacing=0.5)
# ax_two.set_title('Within Regions',font1)
ax.text(1978,970,'Across Regions (Linear fitting)',verticalalignment="top",horizontalalignment="left",fontdict=font1)
ax.text(1973, 970, 'B', verticalalignment="top",
            horizontalalignment="left", fontdict=font3)
set_axis(ax)
fig = plt.gcf()
fig.set_size_inches(15, 8)
fig.savefig(r'chugao\fish_220114_re\figure2\figS6.png',dpi=1000)
plt.show()