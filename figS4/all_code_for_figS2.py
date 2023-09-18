from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from linear_regression import regression
from mk_test import mk_test
from axes_frame import set_axis

font2 = {
    'weight': 'bold',
    'style': 'normal',
    'size': 30,
}


chl_data = pd.read_csv(r'chl_calc\chl_data.csv')
chl_data.replace('--', np.nan, inplace=True)
chl_data.dropna(inplace=True)
chl_data['mean_chl'] = chl_data['mean_chl'].astype(float)

chl_data = chl_data[chl_data['depth'] < 200]

chl_data_lati_mean = chl_data.groupby(['year','month','latitude'])['mean_chl'].mean().reset_index()
chl_data_lati_year_mean = chl_data_lati_mean.groupby(['year','latitude'])['mean_chl'].mean().reset_index()

print(chl_data_lati_year_mean)

'''
mk_test
'''
g = chl_data_lati_year_mean.groupby(['latitude'])
ave_df = pd.DataFrame({'latitude':[], 'concentration':[]})
regr_df = pd.DataFrame({'latitude':[], 'slope':[], 'za':[]})
for idx, single in g:
    ave_df.loc[len(ave_df)] = [float(idx), float(single['mean_chl'].mean())]
    slope, za = mk_test(single['mean_chl'].values.tolist())
    regr_df.loc[len(regr_df)] = [float(idx), float(slope), float(za)]

regr_p_df = regr_df[regr_df['za'] > 1.96]

fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(211)

ax.plot(regr_df['slope'], regr_df['latitude'], c = 'skyblue', label = 'Change rate of chl-a')
ax.vlines(0,0,90,'b')
ax.scatter(regr_p_df['slope'], regr_p_df['latitude'], c = 'r', label = 'Significant points (z > 1.96 | alpha < 0.05)')
font1 = {
    'weight': 'bold',
    'style': 'normal',
    'size': 15,
}
ax.set_xlabel('Change Rate (mg Chl-a / $\mathregular{m^3}$ / year)', c = 'skyblue', fontdict=font1)
ax.set_ylabel('Latitude', fontdict=font1)
ax.set_title('Average Distribution of Chlorophyll-a in the Latitude Direction\nfrom 1993 to 2020 (0.25° latitude resolution)', fontdict=font1)
ax.set_ylim([0,90])
ax.set_xlim([-0.0005,0.002])
handles_ax, labels_ax = ax.get_legend_handles_labels()
set_axis(ax)

ax_two = ax.twiny()
ax_two.plot(ave_df['concentration'], ave_df['latitude'], c = 'orange', label = 'Concentration of chl-a')
ax_two.set_xlabel('Concentration (mg Chl-a / $\mathregular{m^3}$)', c = 'orange',fontdict=font1)
handles_ax_two, labels_ax_two = ax_two.get_legend_handles_labels()
ax_two.set_xticks([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35])
ax_two.set_xticklabels(ax_two.get_xticks(), font1)

ax.legend(handles_ax + handles_ax_two, labels_ax + labels_ax_two, prop=font1, loc = 4)
ax.text(-0.0005 + 0.0025 * 0.02, 90 * 0.93, 'A', font2)




chl_data = pd.read_csv(r'chl_calc\chl_data.csv')
chl_data.replace('--', np.nan, inplace=True)
chl_data.dropna(inplace=True)
chl_data['mean_chl'] = chl_data['mean_chl'].astype(float)

chl_data = chl_data[chl_data['depth'] < 200]
# chl_data = chl_data[chl_data['year'] >= 2007]

chl_data_depth_mean = chl_data.groupby(['year','month','depth'])['mean_chl'].mean().reset_index()
chl_data_depth_year_mean = chl_data_depth_mean.groupby(['year','depth'])['mean_chl'].mean().reset_index()

print(chl_data_depth_year_mean)

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

ax = fig.add_subplot(212)

ax.plot(regr_df['slope'], regr_df['depth'], c = 'skyblue', label = 'Change rate of chl-a')
# ax.vlines(0,0,90,'b')
ax.scatter(regr_p_df['slope'], regr_p_df['depth'], c = 'r', label = 'Significant points (z > 1.96 | alpha < 0.05)')
font1 = {
    'weight': 'bold',
    'style': 'normal',
    'size': 15,
}
ax.set_xlabel('Change Rate (mg Chl-a / $\mathregular{m^3}$ / year)', c = 'skyblue', fontdict=font1)
ax.set_ylabel('Depth (m)', fontdict=font1)
ax.set_title('Average Distribution of Chlorophyll-a in the Depth Direction\nfrom 1993 to 2020 (75 depth levels)', fontdict=font1)
ax.set_xlim([0,0.001])
ax.set_ylim([0,215])
handles_ax, labels_ax = ax.get_legend_handles_labels()
ax.set_xticks([0,0.0002,0.0004,0.0006,0.0008,0.001])
ax.set_xticklabels(ax.get_xticks(), font1)
set_axis(ax)

ax_two = ax.twiny()
ax_two.plot(ave_df['concentration'], ave_df['depth'], c = 'orange', label = 'Concentration of chl-a')
ax_two.set_xlabel('Concentration (mg Chl-a / $\mathregular{m^3}$)', c = 'orange',fontdict=font1)
handles_ax_two, labels_ax_two = ax_two.get_legend_handles_labels()
ax_two.set_xlim([0,0.3])
ax_two.set_xticks([0,0.05,0.1,0.15,0.2,0.25,0.3])
ax_two.set_xticklabels(ax_two.get_xticks(), font1)

ax.legend(handles_ax + handles_ax_two, labels_ax + labels_ax_two, prop=font1, loc = 1)
ax.text(0.001 * 0.02, 215 * 0.93, 'B', font2)
plt.subplots_adjust(wspace=0.1,hspace=0.3)
fig = plt.gcf()
fig.set_size_inches(12, 18)
fig.savefig(r'chl_calc\figS2.jpg', dpi=150)
plt.show()


