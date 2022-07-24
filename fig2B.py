import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

    ax.set_ylim([0,6])
    ax.set_xticklabels(ax.get_xticklabels(),font1)
    ax.set_yticks([1,2,3,4,5])
    ax.set_yticklabels([1,2,3,4,5],font1)
    ax.set_ylabel("Biological nutrition level",font1)
    ax.set_xlabel("Family Shift category",font1)


df = pd.read_csv(r'E:\gephi\data\vertex0.9_node.csv')
ax = sns.violinplot(x='label_5', y='label_2',data = df,bw = 0.4)
set_axis(ax)
fig = plt.gcf()
fig.set_size_inches(10, 10)
fig.savefig(r'chugao\fish_220114_re\figure2\newfig2B.png',dpi=1000)
plt.show()
