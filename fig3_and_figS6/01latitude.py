import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from collections import Counter 
import pickle
from functools import reduce
import seaborn as sns
import scipy
import operator
from axes_frame import set_axis
from mk_test import mk_test
from linear_regression import regression

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

ave_df['range'] = pd.cut(ave_df['latitude'], [-1, 30, 60, 90], labels=[1,2,3])
regr_p_df['range'] = pd.cut(regr_p_df['latitude'], [-1, 30, 60, 90], labels=[1,2,3])

regr_p_df = regr_p_df[['latitude','slope','range']].rename(columns={'slope': 'concentration'})

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
ax.set_yticks([-0.1,0,0.1,0.2,0.3,0.4])
ax.set_yticklabels(['- / -0.0005','0 / 0', '0.1 / 0.0005', '0.2 / 0.001', '0.3 / 0.0015', '0.4 / 0.002'])
ax.set_xlim([-0.5, 2.5])
ax.set_ylim([-0.15, 0.45])
ax.set_ylabel('Concentration / Rate', font1)
ax.set_xlabel('')
ax.legend(title='', prop=font2, loc=4, labelspacing = 0.05)
ax.text(-0.5 + (2.5 + 0.5) * 0.02, -0.15 + (0.45 + 0.15) * 0.9,'A', font1)


final_merge_df = pd.read_csv(r'by_family\final_merge_df.csv',index_col=(0))
# final_merge_df = final_merge_df[final_merge_df['level']!=1]
print(final_merge_df)
# final_merge_df['level'] = final_merge_df['level'].replace([1, 2], 1)
# final_merge_df['level'] = final_merge_df['level'].replace([3, 4, 5], 3)

g = final_merge_df.groupby(['family'])

processed = pd.DataFrame({})

for k, single in g:
    # print(k)
    for i in range(1, 4):
        single['range{}_count_shift'.format(i)] = single['range{}_count'.format(i)].shift(1)
        single['range{}_count_1'.format(i)] = (single['range{}_count'.format(i)] - single['range{}_count_shift'.format(i)]) / single['range{}_count_shift'.format(i)]
        single['range{}_count_diff'.format(i)] = single['range{}_count'.format(i)] - single['range{}_count_shift'.format(i)]
        single.drop(columns=['range{}_count_shift'.format(i)], inplace=True)
    # single.dropna(inplace=True)
    processed = processed.append(single, ignore_index=True)

# plt.hist(processed['range1_count'],bins=1000)
# plt.xlim([-200,200])
# plt.show()

print(processed)


all_data = processed


for ii in range(1, 7):
    processed = all_data
    if ii <= 3:
        r = ii
        processed = processed[processed['range{}_count_diff'.format(r)] > 0]
        # t = processed['range{}_count'.format(r)].median()
        t = np.percentile(processed['range{}_count'.format(r)], 25)
        processed = processed[processed['range{}_count_diff'.format(r)] > t]
        processed = processed[processed['range{}_count_1'.format(r)] > 0.25]
        processed = processed[processed['range{}_count_1'.format(r)] != np.inf]
    else:
        r = ii - 3
        processed = processed[processed['range{}_count_diff'.format(r)] < 0]
        # t = processed['range{}_count'.format(r)].median()
        t = np.percentile(processed['range{}_count'.format(r)], 75)
        processed = processed[processed['range{}_count_diff'.format(r)] < t]
        processed = processed[processed['range{}_count_1'.format(r)] < -0.25]
        processed = processed[processed['range{}_count_1'.format(r)] != np.inf]
    # processed['range3_count'] = processed['range3_count'] / processed['range3_count']

    processed['c'] = processed.groupby(['defined_year'])['range{}_count_diff'.format(r)].transform('count')
    c = processed[['defined_year','c']].drop_duplicates().sort_values(by=['defined_year'])
    # print(c['c'].median())
    # print(np.percentile(c['c'], 25))
    processed = processed[processed['c'] >= np.percentile(c['c'], 5)]

    processed = processed[['family','level','defined_year','count','count_by_year','range{}_count'.format(r),'range{}_count_diff'.format(r),'range{}_count_1'.format(r),'c']]
    # print(processed)

    # info = processed.groupby(['defined_year', 'c', 'level'])['family'].count().reset_index()
    # info['total'] = info.groupby(['defined_year'])['family'].transform('sum')
    # info['perc'] = info['family'] / info['total']
    # print(info)

    # for l in range(1, 6):
    #     sub_info = info[info['level'] == l]
    #     plt.plot(sub_info['defined_year'], sub_info['perc'],label=l)
    # plt.legend()
    # plt.show()
    # exit()
    processed['range{}_count*level'.format(r)] = processed['range{}_count_diff'.format(r)] * processed['level']
    processed['low_high_level'] = pd.cut(processed['level'], [0, 2.5, 6], labels=['low', 'high'])

    # quantity = processed.groupby(['defined_year','level'])['range{}_count_diff'.format(r)].sum().reset_index()
    # print(quantity)

    # color = ['g', 'b', 'orange', 'pink', 'red']
    # slide_quantity = pd.DataFrame({'level':[], 'year':[], 'count':[]})
    # for i in range(1, 4):
    #     year_quantity = quantity[quantity['level'] == i]
    #     for year in range(51 - 14):
    #         info = year_quantity[year_quantity['defined_year'] >= 1970 + year]
    #         info = info[info['defined_year'] < 1985 + year]
    #         slide_quantity = slide_quantity.append({'level':i, 'year':year, 'count':np.mean(np.abs(info['range{}_count_diff'.format(r)]))}, ignore_index=True)

    # slide_quantity['total'] = slide_quantity.groupby(['year'])['count'].transform('sum')
    # slide_quantity['count'] = slide_quantity['count'] / slide_quantity['total']
    # print(slide_quantity)
    # for i in range(1, 4):
    #     y = slide_quantity[slide_quantity['level'] == i]['count']
    #     plt.plot(np.arange(len(y)), y, color[i - 1], label=i)
    # plt.legend()
    # plt.show()

    u = processed.groupby(['defined_year'])['range{}_count*level'.format(r)].sum().reset_index()
    d = processed.groupby(['defined_year'])['range{}_count_diff'.format(r)].sum().reset_index()

    level_count = processed.groupby(['defined_year','low_high_level'])['range{}_count_diff'.format(r)].sum().reset_index()

    low = level_count[level_count['low_high_level'] == 'low']
    high = level_count[level_count['low_high_level'] == 'high']

    y = np.array(u['range{}_count*level'.format(r)] / d['range{}_count_diff'.format(r)])
    xx = []
    yy = []
    win = 15
    for i in range(50 - win + 1):
        # sub_processed = processed[processed['defined_year'] >= 1971 + i]
        # sub_processed = sub_processed[sub_processed['defined_year'] <= 1980 + i]
        # yy.append(sub_processed['range{}_count*level'.format(r)].sum() / sub_processed['range{}_count_diff'.format(r)].sum())

        info1 = u[u['defined_year'] >= 1971 + i]
        info1 = info1[info1['defined_year'] <= 1970 + win + i]
        info2 = d[d['defined_year'] >= 1971 + i]
        info2 = info2[info2['defined_year'] <= 1970 + win + i]
        y = np.array(info1['range{}_count*level'.format(r)] / info2['range{}_count_diff'.format(r)])
        xx.append(1971 + i)
        yy.append(np.mean(y))

    xx = np.array(xx)
    yy = np.array(yy)
    ax = fig.add_subplot(3,3,ii + 3)
    # ax = plt.subplot(4,3,ii)
    # sns.regplot(xx[:30],yy[:30],ax=ax)
    # sns.regplot(xx[30:],yy[30:],color='orange',ax=ax)
    if ii == 2 or ii == 3:
        sns.regplot(xx,yy,color='orange',ax=ax)
        k, R2, P = regression(xx, yy)
        print('=====================================================\n', ii, P, '\n=====================================================')
    else:
        sns.regplot(xx,yy,ax=ax)
        k, R2, P = regression(xx, yy)
        print('=====================================================\n', ii, P, '\n=====================================================')
    ax.set_xlim([1969,2009])
    ax.set_ylim([1.5,3.5])
    set_axis(ax)
    ax.set_xticks([1971,1989,2007])
    ax.set_xticklabels([])
    # ax.set_yticklabels([])
    ax.set_yticks([2,2.5,3])
    ax.set_yticklabels([2,2.5,3])
    # ax.set_xticks([1980,1995,2010])
    # ax.set_xticklabels([1980,1995,2010])
    if ii <= 3:
        ax.text(1969 + (2009 - 1969) * 0.06,1.5 + 2 * 0.9,'B$_{}$'.format(ii), font1)
        # ax.set_title('Region {}'.format(ii,ii), font1)
    else:
        ax.text(1969 + (2009 - 1969) * 0.06,1.5 + 2 * 0.9,'B$_{}$'.format(ii), font1)
    if ii == 1:
        ax.set_ylabel('Average Family\nTrophic Level\n(increasing)', font1)
    if ii == 4:
        ax.set_ylabel('Average Family\nTrophic Level\n(decreasing)', font1)
    if ii >= 4:
        ax.set_xticklabels(['          1970-1984','1988-2002','2006-2020           '])
        ax.set_xlabel('Year', font1)
        
    # yy = []
    # for i in range(50 - win + 1):
    #     info1 = low[low['defined_year'] >= 1971 + i]
    #     info1 = info1[info1['defined_year'] <= 1970 + win + i]
    #     info2 = high[high['defined_year'] >= 1971 + i]
    #     info2 = info2[info2['defined_year'] <= 1970 + win + i]
    #     y = info2['range{}_count_diff'.format(r)].sum() / info1['range{}_count_diff'.format(r)].sum()
    #     yy.append(y)
    # ax = fig.add_subplot(4,3,ii + 6)
    # # ax = plt.subplot(4,3,ii + 6)
    # if ii == 2 or ii == 3:
    #     sns.regplot(xx,yy,color='orange',ax=ax)
    # else:
    #     sns.regplot(xx[:32],yy[:32],ax=ax)
    #     sns.regplot(xx[32:],yy[32:],color='orange',ax=ax)
    
    # set_axis(ax)
    # ax.set_xticks([1980,1995,2010])
    # ax.set_xticklabels([])
    # # ax.set_yticklabels([])
    # ax.set_yticks([0,0.5,1,1.5])
    # ax.set_yticklabels([0,0.5,1,1.5])
    # ax.text(1976,-0.5 + 2.5 * 0.85,'B$_{}$'.format(ii), font1)
    # if ii >= 4:
    #     ax.set_xticklabels(['1975-1990','1990-2005','2005-2020'])
    #     ax.set_xlabel('Year', font1)
    # ax.set_ylim([-0.5,2])

    # if ii == 1:
    #     ax.set_ylabel('Ratio of Species\nat Higher Levels\nto Lower Levels\n(increasing)', font1)
    # if ii == 4:
    #     ax.set_ylabel('Ratio of Species\nat Higher Levels\nto Lower Levels\n(decreasing)', font1)


plt.subplots_adjust(wspace=0.1,hspace=0.1)
fig = plt.gcf()
fig.set_size_inches(20, 15)
fig.savefig(r'fig3\fig3.jpg', dpi=1000)
plt.show()


