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
import pickle
import scipy

def save_variable(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()
    return filename


def load_variavle(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r

f = pd.read_csv(r'family_level_1446.csv')

level = f['level'].values.tolist()
new = f['new_family_index'].values.tolist()
old = f['old_family_index'].values.tolist()
num = f['family_num'].values.tolist()

# pairs : taxonomic level --- old index (in order to uniformation as reserve pairs <new index --- old index>)
reserve = list(zip(level, old))
print(len(reserve))

save_variable(reserve, r'chugao\fish_220114_re\figure2\extra\reserve_level_oldindex.txt')

# Kendall rank correlation analysis between family index and the corresponding taxonimic level
correlation, pvalue = scipy.stats.kendalltau(new, level)
print(correlation, pvalue)
