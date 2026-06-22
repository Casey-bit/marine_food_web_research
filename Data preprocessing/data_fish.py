
import pandas as pd

'''
extract Northern Hemisphere
'''
#total_occurrence = pd.read_csv('data_Occurrence2.csv')
total_occurrence = pd.read_csv('data_Occurrence555.csv')
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


northern_occurrence .to_csv('northern_hemisphere_1970_202016.csv', index=False)

