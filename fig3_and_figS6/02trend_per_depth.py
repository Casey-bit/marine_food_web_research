from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from linear_regression import regression
from mk_test import mk_test
from axes_frame import set_axis

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


chl_data = pd.read_csv(r'chl_calc\chl_data.csv')
chl_data.replace('--', np.nan, inplace=True)
chl_data.dropna(inplace=True)
chl_data['mean_chl'] = chl_data['mean_chl'].astype(float)

# chl_data = chl_data[chl_data['depth'] < 200]
# chl_data = chl_data[chl_data['year'] >= 2007]

chl_data_depth_mean = chl_data.groupby(['year','month','depth'])['mean_chl'].mean().reset_index()
chl_data_depth_year_mean = chl_data_depth_mean.groupby(['year','depth'])['mean_chl'].mean().reset_index()

print(chl_data_depth_year_mean)

'''
regression
'''
# g = chl_data_lati_mean.groupby(['latitude'])
# regr_df = pd.DataFrame({'latitude':[], 'k':[], 'R2':[], 'P':[]})
# for idx, single in g:
#     k, R2, P = regression([y + m / 12 for y, m in list(zip(single['year'], single['month']))], single['mean_chl'])
#     regr_df.loc[len(regr_df)] = [float(idx), float(k), float(R2), float(P)]

# print(regr_df)
# regr_p_df = regr_df[regr_df['P'] < 0.05]

# plt.plot(regr_df['latitude'], regr_df['k'])
# plt.hlines(0,0,90,'b')
# plt.scatter(regr_p_df['latitude'], regr_p_df['k'], c = 'r')
# plt.show()

'''
mk_test
'''
g = chl_data_depth_year_mean.groupby(['depth'])
ave_df = pd.DataFrame({'depth':[], 'concentration':[]})
regr_df = pd.DataFrame({'depth':[], 'slope':[], 'za':[]})
for idx, single in g:
    ave_df.loc[len(ave_df)] = [float(idx), float(single['mean_chl'].mean())]
    slope, za = mk_test(single['mean_chl'].values.tolist())
    regr_df.loc[len(regr_df)] = [float(idx), float(slope), float(za)]

print(regr_df)
regr_p_df = regr_df[regr_df['za'] > 1.96]

ave_df['range'] = pd.cut(ave_df['depth'], [-1, 30, 100, 100000], labels=[1,2,3])
regr_p_df['range'] = pd.cut(regr_p_df['depth'], [-1, 30, 100, 100000], labels=[1,2,3])

regr_p_df = regr_p_df[['depth','slope','range']].rename(columns={'slope': 'concentration'})

ave_df['hue'] = 'Concentration of Chl-a (mg Chl-a / $\mathregular{m^3}$)'
regr_p_df['hue'] = 'Change rate of Chl-a (mg Chl-a / $\mathregular{m^3}$ / year)'

regr_p_df['concentration'] = regr_p_df['concentration'] * 200

ave_df = ave_df.append(regr_p_df, ignore_index = True)

print(ave_df)
print(regr_p_df)

fig = plt.figure()
ax = fig.add_subplot(3,1,1)

sns.violinplot(x='range', y='concentration', hue='hue', data=ave_df, ax=ax, cut=0)
set_axis(ax)
ax.set_xticks([0,1,2])
ax.set_xticklabels(['Range 1', 'Range 2', 'Range 3'])
ax.set_yticks([0,0.1,0.2,0.3])
ax.set_yticklabels(['0 / 0', '0.1 / 0.0005', '0.2 / 0.001', '0.3 / 0.0015'])
ax.set_xlim([-0.5, 2.5])
ax.set_ylim([-0.05, 0.35])
ax.set_ylabel('Concentration / Rate', font1)
ax.set_xlabel('')
ax.legend(title='', prop=font2, loc=1, labelspacing = 0.05)
ax.text(-0.5 + (2.5 + 0.5) * 0.02, -0.05 + (0.35 + 0.05) * 0.9, 'C', font1)

plt.show()

# fig = plt.figure(figsize=(15,8))
# ax = fig.add_subplot(111)

# ax.plot(regr_df['slope'], regr_df['depth'], c = 'skyblue', label = 'slope of chl-a (0.25° resolution)')
# # ax.vlines(0,0,90,'b')
# ax.scatter(regr_p_df['slope'], regr_p_df['depth'], c = 'r', label = 'significant points (z > 1.96 | alpha < 0.05)')
# font1 = {
#     'weight': 'bold',
#     'style': 'normal',
#     'size': 15,
# }
# ax.set_xlabel('slope (chl-a / year)', c = 'skyblue', fontdict=font1)
# ax.set_ylabel('depth', fontdict=font1)
# ax.set_title('MK_test (1993 - 2020)', fontdict=font1)
# ax.set_xlim([0,0.001])
# handles_ax, labels_ax = ax.get_legend_handles_labels()
# ax.set_xticks([0,0.0002,0.0004,0.0006,0.0008,0.001])
# ax.set_xticklabels(ax.get_xticks(), font1)
# set_axis(ax)

# ax_two = ax.twiny()
# ax_two.plot(ave_df['concentration'], ave_df['depth'], c = 'orange', label = 'concentration of chl-a (0.25° resolution)')
# ax_two.set_xlabel('concentration', c = 'orange',fontdict=font1)
# handles_ax_two, labels_ax_two = ax_two.get_legend_handles_labels()
# ax_two.set_xlim([0,0.3])
# ax_two.set_xticks([0,0.05,0.1,0.15,0.2,0.25,0.3])
# ax_two.set_xticklabels(ax_two.get_xticks(), font1)

# plt.legend(handles_ax + handles_ax_two, labels_ax + labels_ax_two, prop=font1, loc = 1)
# plt.show()
