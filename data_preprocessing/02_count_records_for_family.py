'''
 # Copyright (C) 2022-08-09 School of IoT, Jiangnan University Limited
 #
 # All Rights Reserved.
 #
 # Paper: Cascade Shifts of Marine Species in the Northern Hemisphere
 # First author: Zhenkai Wu
 # Corresponding author: Ya Guo, E-mail: guoy@jiangnan.edu.cn
'''
import pandas as pd
from collections import Counter

data = pd.read_csv('E:\\paper\\obis_20220114.csv\\data_Occurrence.csv')

level = 'family' # also can at the taxonomic level of order, genus, etc.

fam = data[level]

pairs = [] # family-num pairs
family_list = [] # nonredundant family
w = dict(Counter(fam)) # the records number of each family

for i in fam:
    if i not in family_list:
        family_list.append(i)
        pairs.append([i,w[i]])
        print(i)

n = [level,'num']
test=pd.DataFrame(columns=n,data=pairs)

test.to_csv('E:\\paper\\obis_20220114.csv\\{}Num.csv'.format(level))
