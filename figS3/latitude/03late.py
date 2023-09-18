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
from mk_test import mk_test

def late_shift(ax):
    family_year_median_df = pd.read_csv(r'cluster\family_year_median_df_over2000.csv', index_col=(0))

    g = family_year_median_df.groupby(['family'])

    mk_test_res = pd.DataFrame({'family':[], 'slope':[], 'za':[]})

    for k, single in g:
        # data = []
        # for year in range(51 - 9):
        #     sub = single[single['year'] >= 1970 + year]
        #     sub = sub[sub['year'] < 1980 + year]
        #     if len(sub) > 0:
        #         sub['count_by_year'] = sub['count_by_year'] / sub['count_by_year'].sum()
        #         sub['median'] = sub['median'] * sub['count_by_year']
        #         data.append(sub['median'].sum())

        # slope, za = mk_test(data)
        slope, za = mk_test(single['median'].values.tolist())
        # print(slope, za)
        mk_test_res.loc[len(mk_test_res)] = [k, float(slope), float(za)]

    mk_test_res = mk_test_res[mk_test_res['za'] > 1.96]

    print(mk_test_res)

    family_year_median_df = pd.merge(family_year_median_df, mk_test_res, on=['family'])

    color1 = [(236/255, 95/255, 116/255, 0.8), (255/255, 111/255, 105/255, 0.8), (160/255, 64/255, 160/255, 0.8),
                (205/255, 62/255, 205/255, 0.8), (46/255, 117/255, 182/255, 0.8), (52/255, 152/255, 219/255, 0.8)]
    color2 = [(236/255, 95/255, 116/255, 0.2), (255/255, 111/255, 105/255, 0.2), (160/255, 64/255, 160/255, 0.2),
                (205/255, 62/255, 205/255, 0.2), (46/255, 117/255, 182/255, 0.2), (52/255, 152/255, 219/255, 0.2)]
    label = ['Family shift northward (0-30°N)', 'Family shift southward (0-30°N) (lack of significance)', 'Family shift northward (30°N-60°N)',
             'Family shift southward (30°N-60°N)', 'Family shift northward (60°N-90°N)', 'Family shift southward (60°N-90°N)']
    for i in range(1, 4):
        sub = family_year_median_df[family_year_median_df['belonging'] == i]

        family_year_median_df_1 = sub[sub['slope'] > 0]
        family_year_median_df_2 = sub[sub['slope'] < 0]

        res_1 = family_year_median_df_1.groupby(['year'])['median'].mean().reset_index()
        std_1 = family_year_median_df_1.groupby(['year'])['median'].std(ddof=1).reset_index()
        res_2 = family_year_median_df_2.groupby(['year'])['median'].mean().reset_index()
        std_2 = family_year_median_df_2.groupby(['year'])['median'].std(ddof=1).reset_index()

        data1 = []
        std1 = []
        for year in range(51 - 9):
            sub = res_1[res_1['year'] >= 1970 + year]
            sub = sub[sub['year'] < 1980 + year]
            data1.append(sub['median'].mean())

            sub1 = std_1[std_1['year'] >= 1970 + year]
            sub1 = sub1[sub1['year'] < 1980 + year]
            std1.append(sub1['median'].mean())
        data2 = []
        std2 = []
        for year in range(51 - 9):
            sub = res_2[res_2['year'] >= 1970 + year]
            sub = sub[sub['year'] < 1980 + year]
            data2.append(sub['median'].mean())

            sub2 = std_2[std_2['year'] >= 1970 + year]
            sub2 = sub2[sub2['year'] < 1980 + year]
            std2.append(sub2['median'].mean())
        if len(family_year_median_df_1['family'].drop_duplicates()) > 2:
            ax.plot(data1, color = color1[2 * (i - 1)], linewidth=3, label=label[2 * (i - 1)])
            ax.fill_between(np.arange(
                0, 51 - 9, 1), np.array(data1) - np.array(std1) / 2, np.array(data1) + np.array(std1) / 2, facecolor=color2[2 * (i - 1)])
        else:
            ax.plot([], color = color1[2 * (i - 1)], linewidth=3, label=label[2 * (i - 1)])
        if len(family_year_median_df_2['family'].drop_duplicates()) > 2:
            ax.plot(data2, '--',color = color1[2 * (i - 1) + 1], linewidth=3, label=label[2 * (i - 1) + 1])
            ax.fill_between(np.arange(
                0, 51 - 9, 1), np.array(data2) - np.array(std2) / 2, np.array(data2) + np.array(std2) / 2,facecolor=color2[2 * (i - 1) + 1])
        else:
            ax.plot([], '--', color = color1[2 * (i - 1) + 1], linewidth=3, label=label[2 * (i - 1) + 1])
    font2 = {
        'weight': 'bold',
        'style': 'normal',
        'size': 12,
    }
    ax.legend(loc=4,prop=font2, fontsize=12, markerscale=0.5, columnspacing=0.5)
