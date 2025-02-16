import pandas as pd

#https://obis.org/data/access/
#Data download date: October 2022.
data = pd.read_csv('obis_20221006.csv',usecols=(2,3,16,17,29,32,41,46,51,56,61,83,123))  
data = data.dropna(subset=['eventdate'])
data = data[data['eventdate'].str[0:4].str.isdigit()]

data_Occurrence = data[data['basisofrecord']=='Occurrence']
data_Occurrence.to_csv('data_Occurrence.csv')
