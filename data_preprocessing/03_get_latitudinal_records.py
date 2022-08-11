'''
 # Copyright (C) 2022-08-09 School of IoT, Jiangnan University Limited
 #
 # All Rights Reserved.
 #
 # Paper: Successive Relocation of Marine Species in the Northern Hemisphere from 1970 to 2020
 # First author: Zhenkai Wu
 # Corresponding author: Ya Guo, E-mail: guoy@jiangnan.edu.cn
'''
import pandas as pd
import pickle

def save_variable(v,filename):
    f=open(filename,'wb')
    pickle.dump(v,f)
    f.close()
    return filename

def load_variavle(filename):
    f=open(filename,'rb')
    r=pickle.load(f)
    f.close()
    return r

type = 'family' # also can at the taxonomic level of order, genus, etc.

dffamily = pd.read_csv('E:\\paper\\obis_20220114.csv\\{}Num.csv'.format(type))
dfdata = pd.read_csv('E:\\paper\\obis_20220114.csv\\data_Occurrence.csv')

family = dffamily[type].values.tolist()

date = dfdata['eventdate'].values.tolist()
lat = dfdata['decimallatitude'].values.tolist()
lon = dfdata['decimallongitude'].values.tolist()
famdata = dfdata[type].values.tolist()

start = 1970
end = 2020
yearTotal = end - start + 1
familyNum = len(family)

# countByYear[y][i] : all latitudinal records of the famliy (index i) in year (index y)
countByYear = [[[] for family in range(familyNum)] for year in range(yearTotal)]

for var in range(familyNum):
    variable = family[var]
    print(variable, var)
    for r in range(len(famdata)):
        if start <= int(date[r][0:4]) <= end and famdata[r] == variable:
            year = int(date[r][0:4]) - start
            countByYear[year][var].append(lat[r])

save_variable(countByYear,'AA_fishCount_ph\\count\\latitudeByYear_occurrence_{}_{}_{}_{}.txt'.format(start,end,familyNum,type))
