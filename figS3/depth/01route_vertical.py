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
2.1970-2020
'''
northern['year'] = northern['eventdate'].astype(str).str.slice(0,4)
northern.drop(columns=['eventdate'], inplace=True)
northern = northern[northern['year'].astype(int) >= 1970]
northern = northern[northern['year'].astype(int) <= 2020]

northern = northern[northern['bathymetry'] >= 0].reset_index().drop(columns=['index'])
# northern = northern[northern['bathymetry'] < 200].reset_index().drop(columns=['index'])
# 14351128  13094558
'''
3.count by family
'''
northern['count_by_family'] = northern.groupby(['family'])['bathymetry'].transform('count')
northern.drop(columns=['Unnamed: 0', 'basisofrecord'], inplace=True)
northern.dropna(subset=['count_by_family'], inplace=True)
northern = northern.sort_values(by=['count_by_family'], ascending=False).reset_index().drop(columns=['index'])
'''
extract <100
'''
northern = northern[northern['count_by_family'].astype(int) >= 100].reset_index().drop(columns=['index'])
'''
judge region
'''
northern['year'] = northern['year'].astype(int)
for i in range(3):
    if i == 0:
        northern['range' + str(i + 1)] = pd.cut(northern['bathymetry'], [0, 30], labels=['(0,50]'])
    elif i == 1:
        northern['range' + str(i + 1)] = pd.cut(northern['bathymetry'], [30, 100], labels=['(50,100]'])
    elif i == 2:
        northern['range' + str(i + 1)] = pd.cut(northern['bathymetry'], [100, 10000], labels=['(100,10000]'])
    sub = northern[northern['year'] < 1990]
    c = sub.groupby(['family'])['range' + str(i + 1)].count().reset_index().rename(columns={'range' + str(i + 1): 'range{}_count'.format(i + 1)})
    northern = pd.merge(northern, c, on=['family'])
    # northern['range{}_count'.format(i + 1)] = northern.groupby(['family'])['range' + str(i + 1)].transform('count')
    northern.drop(columns=['range' + str(i + 1)], inplace=True)
    northern.fillna(value={'range{}_count'.format(i + 1): 0}, inplace=True)

family_belonging_df = northern.groupby(['family','count_by_family','range1_count','range2_count','range3_count'])['bathymetry'].count().reset_index().drop(columns=['bathymetry'])


family_belonging_df['belonging'] = family_belonging_df[['range1_count','range2_count','range3_count']].idxmax(axis=1)
family_belonging_df['belonging'] = family_belonging_df['belonging'].astype(str).str.slice(5,6).astype(int)
print(family_belonging_df)

'''
median
'''
family_year_median_df = northern.groupby(['family', 'year'])['bathymetry'].median().reset_index().rename(columns={"bathymetry": "median"})

family_belonging_df = family_belonging_df[['family', 'belonging']]
family_year_median_df = pd.merge(family_year_median_df, family_belonging_df, on=['family'])


# family_year_median_df = family_year_median_df[family_year_median_df['belonging'] == 2]
family_year_median_df.to_csv(r'cluster\family_year_median_df_depth_early1990.csv')

print(family_year_median_df)
