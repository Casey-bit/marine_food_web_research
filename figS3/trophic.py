import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from collections import Counter  
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

f = pd.read_csv(r'by_species\family\final_merge_df.csv')
f = f[['family','level']].drop_duplicates()

level = f['level'].values.tolist()

y = [1,2,3,4,5]
width = [len([i for i in level if i == 1]),
        len([i for i in level if i == 2]),
        len([i for i in level if i == 3]),
        len([i for i in level if i == 4]),
        len([i for i in level if i == 5])]

ax = plt.subplot()
ax.barh(y = y, width=width, alpha = 0.5)
t = ['Algae, plants, and detritus', 'Omnivores,\nherbivores,\nand detritivores',
    'Mid-level carnivores\n(predators with target of zooplankton or mini-size species)',
    'High-level carnivores\n(mild or mid-size piscivores)',
    'Top predators\n(fierce or large-size piscivores)']

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

for a, b in zip(y, width):
    ax.text(b + 2, a, t[a - 1], ha = 'left', va = 'center', fontdict=font2)
    ax.text(b - 5, a, b, ha = 'right', va = 'center', fontdict=font2, color=(255/255, 26/255, 128/255))

ax.set_xlim([0,500])
ax.set_xticks([100 * i for i in range(5)])
ax.set_xticklabels([100 * i for i in range(5)], font1)
ax.set_xlabel('Number of Families',font1)
ax.set_yticks(y)
ax.set_yticklabels(y, font1)
ax.set_ylabel('Trophic Level',font1)
set_axis(ax)
fig = plt.gcf()
fig.set_size_inches(10, 6)
fig.savefig(r'figS\figS4.jpg',dpi=150)
plt.show()
