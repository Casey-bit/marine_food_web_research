import pandas as pd
import numpy as np
from mk_test import mk_test
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

df = pd.read_csv(r'E:\gephi\data\vertex0.9_cluster_latitude_230730.csv')
node = [] # 非孤立节点
df['source'] = df['source'].astype(int)
df['target'] = df['target'].astype(int)

for elem in df['source']:
    if not elem in node:
        node.append(elem)

for elem in df['target']:
    if not elem in node:
        node.append(elem)
print(len(node))
reserve = load_variavle(r'fig3\reserve.txt')
print(len(reserve))
median = load_variavle(r'fig3\median_denoising.txt')

mk_test_res = pd.DataFrame({'family':[], 'slope':[], 'za':[], 'describe':[], 'mode':[]})

for n, o in reserve:
    if n in node:
        slope, za = mk_test(median[o])
        if za > 1.96:
            if slope > 0:
                mk_test_res.loc[len(mk_test_res)] = [n, float(slope), float(za), 'Northward', 1]
            if slope < 0:
                mk_test_res.loc[len(mk_test_res)] = [n, float(slope), float(za), 'Southward', 2]
        else:
            mk_test_res.loc[len(mk_test_res)] = [n, float(slope), float(za), 'Mixed', 3]

for i in range(1, 4):
    print(len(mk_test_res[mk_test_res['mode'] == i]))

node_df = pd.read_csv(r'E:\gephi\data\vertex0.9_node_latitude_230730.csv')
node_df = node_df[['id', 'level']]

mk_test_res = pd.merge(mk_test_res, node_df, left_on=['family'], right_on=['id'])
print(mk_test_res)
mk_test_res_2 = mk_test_res.copy()
mk_test_res_2['describe'] = 'Total'
mk_test_res_2['mode'] = 0

mk_test_res = mk_test_res.append(mk_test_res_2, ignore_index = True)
print(mk_test_res)