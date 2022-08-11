import pandas as pd

data = pd.read_csv('E:\\paper\\obis_20220114.csv\\obis_20220114.csv',usecols=(2,3,16,17,29,32,41,46,51,56,61,83,123))
data = data.dropna(subset=['eventdate'])
data = data[data['eventdate'].str[0:4].str.isdigit()]

data_Occurrence = data[data['basisofrecord']=='Occurrence']
data_Occurrence.to_csv('E:\\paper\\obis_20220114.csv\\data_Occurrence.csv')