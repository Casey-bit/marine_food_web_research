'''
 # Copyright (C) 2022-08-09 School of IoT, Jiangnan University Limited
 #
 # All Rights Reserved.
 #
 # Paper: Successive Relocation of Marine Species in the Northern Hemisphere from 1970 to 2020
 # First author: Zhenkai Wu
 # Corresponding author: Ya Guo, E-mail: guoy@jiangnan.edu.cn
'''
from netCDF4 import Dataset
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
font1 = {
        'weight': 'bold',
        'style':'normal',
        'size': 15,
        }
font2 = {
        'weight': 'bold',
        'style':'normal',
        'size': 12,
        }

def set_axis(ax):

    ax.grid(False)
    ax.set_xticklabels(ax.get_xticklabels())
    # ax.set_xticks([0,0.15,0.3])
    # ax.set_xticklabels(['$\mathregular{_{-0.03}}$\n0','$\mathregular{_{0}}$\n0.15','$\mathregular{_{+0.03}}$\n0.3'], font1)
    # ax.set_xlabel('Total chlorophyll-a concentration (mg chl-a / $\mathregular{m^3}$)', font1, loc = 'right')
    # l = ax.legend(prop = font2,loc = 4,title = 'chlorophyll-a concentration distribution')
    # l.get_title().set_fontsize(12)
    # l.get_title().set_fontweight('bold')
    # ax.grid('x')
    ax.spines['top'].set_linewidth('2.0')
    ax.spines['bottom'].set_linewidth('2.0')
    ax.spines['left'].set_linewidth('2.0')
    ax.spines['right'].set_linewidth('2.0')
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    for tickline in ax.xaxis.get_ticklines():
        tickline.set_visible(True)
    for tickline in ax.yaxis.get_ticklines():
        tickline.set_visible(True)
    for tickline in ax.yaxis.get_minorticklines():
        tickline.set_visible(True)
    ax.tick_params(which="major",
                    length=15,width = 2.0,
                    colors = "black",
                    direction = 'in',
                    tick1On = False,tick2On = False)
    ax.tick_params(which="minor",
                    length=5,width = 1.0,
                    colors = "black",
                    direction = 'in',
                    tick1On = False)                 
    # ax.tick_params(which = "minor",
    #                 length = 5, width = 1.0,
    #                 labelsize=10,labelcolor = "0.25")
    

def sub_one(ax):
    variable = 'tot'
    # dict_keys(['talk', 'dic', 'spco2', 'sfco2', 'ph_total', 
    # hco3', 'co3', 'co2', 'revelle_factor', 'omega_ca', 'omega_ar',
    #  'dic_uncert', 'spco2_uncert', 'ph_total_uncert', 'omega_ca_uncert', 
    # 'omega_ar_uncert', 'temperature', 'salinity', 'talk_uncert', 'sfco2_uncert', 
    # 'fgco2', 'fgco2_global', 'area', 'lat', 'lon', 'time'])

    nc = Dataset('data\\day19980101.R2017.nc4')
    long_tot = nc.variables['lon'][:]  
    lati_tot = nc.variables['lat'][:]

    aveYearLatitude = load_variavle('chugao\\tot_influence_fish\\aveYearLatitude_{}.txt'.format(variable))

    aveplot = np.array([0 for lat in range(aveYearLatitude.shape[1])],dtype=float)
    for j in range(aveYearLatitude.shape[0]):
        aveplot += aveYearLatitude[j]
    aveplot /= aveYearLatitude.shape[0]

    ax.set_xlim([0,2])
    # for i in range(len(aveplot)):
    #     if np.power(aveplot[i],10) * 12000 >= (np.power(0.32,10) * 15000):
    #         aveplot[i] = np.power(aveplot[i],10) * 15000
    #     else:
    #         aveplot[i] = (np.power(0.32,10) * 15000) * aveplot[i] / 0.32
    # ax.barh(y = lati_tot, width = aveplot, color = (150/255,220/255,180/255,1),zorder = -100)
    # aveplotover = aveplot - np.power(0.335,10) * 15000
    # aveplotover[aveplotover <= 0] = np.nan
    # ax.barh(y = lati_tot, width = aveplotover,left = [np.power(0.335,10) * 15000 for i in range(len(lati_tot))], color = 'red',zorder = -100)
    ax.invert_xaxis()

    aveplot = np.array([[0 for lat in range(aveYearLatitude.shape[1])] for curve in range(2)],dtype=float)
    style = ['-','--']
    for i in range(2):
        for j in range(9):
            aveplot[i] += aveYearLatitude[9 * i + j]
        aveplot[i] /= 9
        ax.plot(aveplot[i],lati_tot,style[i], color = (150/255,220/255,180/255,0.8),linewidth = 3,label = 'Chl-a (' + str(1998 + 9 * i) + " ~ " + str(1998 + 9 * i + 8) + ')')
    # ax.plot(aveplot[1] - aveplot[0],lati_tot,style[i], color = 'red',alpha = 0.5,linewidth = 3,label = 'change')
    ax.barh(y = lati_tot, width = (aveplot[1] - aveplot[0]) * 5, left = [0.15 for i in range(len(lati_tot))], color = 'gray',alpha = 0.5,label = 'Changes in Chl-a')
    # ax.scatter(aveplot[0, 215 + np.argmax(aveplot[0, 215:233])], lati_tot[215 + np.argmax(aveplot[0, 215:233])],s = 50, marker = '*', color = 'red',label = 'Peak point of Chl-a')
    # ax.scatter(aveplot[1, 215 + np.argmax(aveplot[1, 215:233])], lati_tot[215 + np.argmax(aveplot[1, 215:233])],s = 50, marker = '*', color = 'red')

    # ax.scatter(aveplot[0, 190 + np.argmax(aveplot[0, 190:215])], lati_tot[190 + np.argmax(aveplot[0, 190:215])],s = 50, marker = '*', color = 'red')
    # ax.scatter(aveplot[1, 190 + np.argmax(aveplot[1, 190:215])], lati_tot[190 + np.argmax(aveplot[1, 190:215])],s = 50, marker = '*', color = 'red')
    ax.plot([0,0.5],[75,75],linewidth = 2,color='black')
    ax.plot([0.15,0.15],[75,73],linewidth = 2,color='black')
    ax.plot([0.3,0.3],[75,73],linewidth = 2,color='black')
    ax.plot([0.15,0.15],[73,0],linewidth = 0.5,color='gray')
    ax.plot([0.3,0.3],[73,0],linewidth = 0.5,color='gray')
    ax.text(0.005,75,'0',color=(150/255,220/255,180/255),verticalalignment="bottom",horizontalalignment="right",fontdict=font1)
    ax.text(0.155,75,'0.15',color=(150/255,220/255,180/255),verticalalignment="bottom",horizontalalignment="right",fontdict=font1)
    ax.text(0.305,75,'0.3',color=(150/255,220/255,180/255),verticalalignment="bottom",horizontalalignment="right",fontdict=font1)
    ax.text(0.005,74.85,'-0.03',color = 'gray',verticalalignment="top",horizontalalignment="right",fontdict=font2)
    ax.text(0.155,74.85,'0',color = 'gray',verticalalignment="top",horizontalalignment="right",fontdict=font2)
    ax.text(0.305,74.85,'+0.03',color = 'gray',verticalalignment="top",horizontalalignment="right",fontdict=font2)
    ax.text(0.005,77,'Total chlorophyll-a concentration (mg chl-a / $\mathregular{m^3}$)',verticalalignment="bottom",horizontalalignment="right",fontdict=font1)
    set_axis(ax)
    ax.set_zorder(-10)
    handles,labels = ax.get_legend_handles_labels()
    return handles,labels
