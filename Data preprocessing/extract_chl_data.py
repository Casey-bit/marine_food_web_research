from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from statsmodels.formula.api import ols

path = []
filePath = 'cmems_mod_glo_bgc_my_0.25_P1M-m'   #https://marine.copernicus.eu/
for dirpath, dirnames, filenames in os.walk(filePath):
    path = path + [os.path.join(dirpath, names) for names in filenames]



for file in path:
    nc = Dataset(file)
    year = int(file[84:88])
    month = int(file[88:90])
    long = nc.variables['longitude'][:]
    # [-180.   -179.75 -179.5  ...  179.25  179.5   179.75]
    lati = nc.variables['latitude'][:]
    # [-80.   -79.75 -79.5  -79.25 -79.  ...  89.    89.25  89.5   89.75 90.  ]
    depth = nc.variables['depth'][:]
    data = nc.variables['chl'][0,:]
    # print(data.shape) # (75, 681, 1440)
    
    for d in range(len(depth)):        
        year_ls = []
        month_ls = []
        depth_ls = []
        lat_ls = []
        mean_chl_ls = []

        for north_lati_index in range(320, 681):
            year_ls.append(year)
            month_ls.append(month)
            depth_ls.append(depth[d])
            lat_ls.append(lati[north_lati_index])
            mean_chl_ls.append(np.mean(data[d, north_lati_index]))
        
        sub_res_df = pd.DataFrame({})
        sub_res_df['year'] = year_ls
        sub_res_df['month'] = month_ls
        sub_res_df['depth'] = depth_ls
        sub_res_df['latitude'] = lat_ls
        sub_res_df['mean_chl'] = mean_chl_ls

        if year == 1993 and month == 1 and d == 0:
            sub_res_df.to_csv(r'chl_calc\chl_data.csv', index=False)
        else:
            sub_res_df.to_csv(r'chl_calc\chl_data.csv', mode='a', header=False, index=False)

    print(year, month)
