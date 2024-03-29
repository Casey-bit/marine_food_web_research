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

'''
1. extract Northern Hemisphere
'''
total_occurrence = pd.read_csv('E:\\paper\\obis_20221006\\data_Occurrence.csv')
northern = total_occurrence[total_occurrence['decimallatitude'] > 0]
'''
1970-2020
'''
northern['year'] = northern['eventdate'].astype(str).str.slice(0,4)
northern.drop(columns=['eventdate'], inplace=True)
northern = northern[northern['year'].astype(int) >= 1970]
northern = northern[northern['year'].astype(int) <= 2020]

northern = northern[northern['bathymetry'] >= 0].reset_index().drop(columns=['index'])

'''
count by family
'''
northern['count_by_family'] = northern.groupby(['family'])['bathymetry'].transform('count')
northern.drop(columns=['Unnamed: 0', 'basisofrecord'], inplace=True)
northern.dropna(subset=['count_by_family'], inplace=True)
northern = northern.sort_values(by=['count_by_family'], ascending=False).reset_index().drop(columns=['index'])

'''
count by year
'''
northern['count_by_year'] = northern.groupby(['family','year'])['bathymetry'].transform('count')

'''
exclude families with records less than 100 
'''
northern = northern[northern['count_by_family'].astype(int) >= 100].reset_index().drop(columns=['index'])
northern.to_csv(r'by_family\method_2\new230729\northern_d.csv')
'''
ranges' weight
data correction
'''
northern['range'] = pd.cut(northern['bathymetry'], [0,12.5,25,37.5,50,
                                                    62.5,75,87.5,100,112.5,
                                                    125,137.5,150,162.5,175,
                                                    187.5,200,10000], labels=np.arange(1,18,1))
year_range_count = northern.groupby(['year','range'])['bathymetry'].count().reset_index()
year_range_count['range_mean'] = year_range_count.groupby(['range'])['bathymetry'].transform('mean')
year_range_count['corr'] = year_range_count['range_mean'] / year_range_count['bathymetry']
year_range_count['corr'][year_range_count['corr'] > 1e8] = 0
year_range_count = year_range_count[['year','range','corr']]

'''
judge belonging range
'''
for i in range(3):
    if i == 0:
        northern['range' + str(i + 1)] = pd.cut(northern['bathymetry'], [0, 30], labels=['(0,50]'])
    elif i == 1:
        northern['range' + str(i + 1)] = pd.cut(northern['bathymetry'], [30, 100], labels=['(50,100]'])
    elif i == 2:
        northern['range' + str(i + 1)] = pd.cut(northern['bathymetry'], [100, 10000], labels=['(100,10000]'])
    northern['range{}_count'.format(i + 1)] = northern.groupby(['family','year'])['range' + str(i + 1)].transform('count')
    northern.drop(columns=['range' + str(i + 1)], inplace=True)
    northern.fillna(value={'range{}_count'.format(i + 1): 0}, inplace=True)

family_belonging_df = northern.groupby(['family','year','count_by_year','range1_count','range2_count','range3_count'])['decimallatitude'].count().reset_index().drop(columns=['decimallatitude'])


family_belonging_df['belonging'] = family_belonging_df[['range1_count','range2_count','range3_count']].idxmax(axis=1)
family_belonging_df['belonging'] = family_belonging_df['belonging'].astype(str).str.slice(5,6).astype(int)
'''
get yearly median
'''
family_year_median_df = northern.groupby(['family', 'year', 'range'])['bathymetry'].median().reset_index().rename(columns={"bathymetry": "median"})
family_year_median_df.dropna(inplace=True)
family_year_median_df = family_year_median_df.merge(year_range_count, on=['year','range'])
family_year_median_df['median_up'] = family_year_median_df['median'] * family_year_median_df['corr']

family_year_corrsum = family_year_median_df.groupby(['family','year'])['corr'].sum().reset_index()
family_year_medupsum = family_year_median_df.groupby(['family','year'])['median_up'].sum().reset_index()

family_year_median_df = pd.merge(family_year_corrsum, family_year_medupsum, on=['family','year'])
family_year_median_df['median'] = family_year_median_df['median_up'] / family_year_median_df['corr']
family_year_median_df = family_year_median_df[['family','year','median']]

family_belonging_df = pd.merge(family_belonging_df, family_year_median_df, left_on=['family','year'], right_on=['family','year'])


g = family_belonging_df.groupby(['family'])
merge_df_processed = pd.DataFrame({})
for k, single_family in g:

    single_family_df = pd.DataFrame(single_family)

    '''
    judge belonging year = ceil[(pre + next) / 2]
    '''
    single_family_df['year'] = single_family_df['year'].astype(int)
    single_family_df['year_shift'] = single_family_df['year'].shift(1)
    single_family_df['defined_year'] = np.ceil((single_family_df['year'] + single_family_df['year_shift']) / 2)
    # single_family_df['defined_year'] = single_family_df['year']
    single_family_df.drop(columns=['year_shift'], inplace=True)
    single_family_df = single_family_df.loc[:, ['family','year','defined_year','median','count_by_year','range1_count','range2_count','range3_count','belonging']]

    single_family_df = single_family_df.fillna(method='pad', axis=1)
    merge_df_processed = merge_df_processed.append(single_family_df, ignore_index=True)


reserved_family = merge_df_processed['family'].drop_duplicates().reset_index().drop(columns=['index'])
reserved_family['reserved'] = 'True'
print(reserved_family)

northern = pd.merge(northern, reserved_family, left_on='family', right_on='family')
# print(northern)


family_count_df = northern.groupby(['family'])['bathymetry'].count().reset_index().rename(columns={"bathymetry": "count"})
family_count_df = family_count_df.sort_values(by=['count'], ascending=False).reset_index().drop(columns=['index'])
family_count_df['index'] = family_count_df['count'].rank(ascending=False)

final_merge_df = pd.merge(merge_df_processed, family_count_df, left_on='family', right_on='family')
'''
trophic level
'''
family_level = pd.read_csv(r'by_species\level\function_group_2.csv',usecols=(8,13))
family_level.drop_duplicates(subset=['family'],inplace=True)
family_level.dropna(inplace=True)
final_merge_df = pd.merge(final_merge_df, family_level, left_on='family', right_on='family')

final_merge_df = final_merge_df[['family','level','defined_year','median','count','count_by_year','range1_count','range2_count','range3_count']]
final_merge_df = final_merge_df.loc[:, ['family','level','defined_year','median','count','count_by_year','range1_count','range2_count','range3_count']]
# final_merge_df = final_merge_df[final_merge_df['level'] != 1]
final_merge_df.to_csv(r'by_family\depth\final_merge_df_0_30_100_10000.csv')

print(final_merge_df)
