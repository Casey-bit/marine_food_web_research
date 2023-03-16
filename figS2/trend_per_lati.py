from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from linear_regression import regression
from mk_test import mk_test
from axes_frame import set_axis


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
ax = fig.add_subplot(111)

ax.plot(regr_df['slope'], regr_df['latitude'], c = 'skyblue', label = 'Change rate of chl-a')
ax.vlines(0,0,90,'b')
ax.scatter(regr_p_df['slope'], regr_p_df['latitude'], c = 'r', label = 'Significant points (z > 1.96 | alpha < 0.05)')
font1 = {
    'weight': 'bold',
    'style': 'normal',
    'size': 15,
}
ax.set_xlabel('slope (chl-a / year)', c = 'skyblue', fontdict=font1)
ax.set_ylabel('latitude', fontdict=font1)
ax.set_title('Average Distribution of Chlorophyll-a in the Latitude Direction from 1993 to 2020 (0.25° latitude resolution)', fontdict=font1)
ax.set_ylim([0,90])
handles_ax, labels_ax = ax.get_legend_handles_labels()
set_axis(ax)

ax_two = ax.twiny()
ax_two.plot(ave_df['concentration'], ave_df['latitude'], c = 'orange', label = 'Concentration of chl-a')
ax_two.set_xlabel('concentration', c = 'orange',fontdict=font1)
handles_ax_two, labels_ax_two = ax_two.get_legend_handles_labels()
ax_two.set_xticks([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35])
ax_two.set_xticklabels(ax_two.get_xticks(), font1)

plt.legend(handles_ax + handles_ax_two, labels_ax + labels_ax_two, prop=font1, loc = 4)
plt.show()
