import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

'''
extract Northern Hemisphere
'''
total_occurrence = pd.read_csv(r'E:\\paper\\obis_20221006\\data_Occurrence.csv')
northern_occurrence = total_occurrence[total_occurrence['decimallatitude'] > 0]
'''
1970-2020
'''
northern_occurrence['year'] = northern_occurrence['eventdate'].astype(str).str.slice(0,4)
northern_occurrence.drop(columns=['eventdate','basisofrecord','Unnamed: 0'], inplace=True)
northern_occurrence = northern_occurrence[northern_occurrence['year'].astype(int) >= 1970]
northern_occurrence = northern_occurrence[northern_occurrence['year'].astype(int) <= 2020]
'''
bathymetry > 0
'''
northern_occurrence = northern_occurrence[northern_occurrence['bathymetry'] > 0]
'''
count by species
'''
northern_occurrence['count'] = northern_occurrence.groupby(['kingdom','phylum','order','family','genus','species'],dropna=False)['decimallatitude'].transform('count')
'''
count by year
'''
northern_occurrence['count_by_year'] = northern_occurrence.groupby(['kingdom','phylum','order','family','genus','species','year'],dropna=False)['decimallatitude'].transform('count')
'''
exclude species's yearly records which less than 10
'''
northern_occurrence = northern_occurrence[northern_occurrence['count_by_year'].astype(int) >= 10].reset_index().drop(columns=['index'])
'''
exclude species whose records distributed less than 10 years
'''
northern_occurrence['has_data_yearnum'] = northern_occurrence.groupby(['kingdom','phylum','order','family','genus','species'],dropna=False)['year'].transform('nunique')
northern_occurrence = northern_occurrence[northern_occurrence['has_data_yearnum'].astype(int) >= 10].reset_index().drop(columns=['index'])

by_species = northern_occurrence.groupby(['kingdom','phylum','order','family','genus','species'],dropna=False)['decimallatitude'].count().reset_index().rename(columns={'decimallatitude':'count_by_species'})

by_species = by_species.sort_values(by=['count_by_species'], ascending=False).reset_index().drop(columns=['index'])
print(by_species)
by_species.to_csv(r'by_species\level\species_list.csv')
