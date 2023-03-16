import pandas as pd
import numpy as np
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

median_df = pd.read_csv(r'fig2\latitude\family_year_median_df.csv',index_col=(0))
print(median_df)
family = median_df[['family','level']].drop_duplicates()
family.reset_index(inplace=True)
fam = family['family'].drop_duplicates()
fam.to_csv(r'fig2\latitude\FamilyNum_1.csv')

median_array = np.array([[np.nan for year in range(51)] for family in range(len(family))]) 
count_array = np.array([0.0 for family in range(len(family))])

g = median_df.groupby(['family'])

i_f = 0
for k, single in g:
    print(i_f, len(family))
    y = single['year'].values.tolist()
    m = single['median'].values.tolist()
    count_array[i_f] = (single['count_by_family'].values.tolist())[0]
    for i in range(len(y)):
        median_array[i_f, y[i] - 1970] = m[i]
    i_f += 1  

save_variable(median_array, r'fig2\latitude\medianPoint.txt')
save_variable(count_array, r'fig2\latitude\countArray.txt')
print(count_array)

reserve = load_variavle(r'fig2\latitude\reserve.txt')

family['label'] = np.nan
print(family)
for n, o in reserve:
    print(o)
    family['label'].loc[[o]] = n
family.dropna(inplace=True)
family.to_csv(r'E:\gephi\data\vertex0.9_node_230125.csv')
print(family)
