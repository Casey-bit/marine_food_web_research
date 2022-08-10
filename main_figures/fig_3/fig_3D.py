import numpy as np
import pickle
from scipy import stats
from tsmoothie.smoother import LowessSmoother

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

def set_axis(ax):
    ax.grid(False)
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
    ax.tick_params(which="major",
                    length=15,width = 2.0,
                    colors = "black",
                    direction = 'in',
                    tick2On = False)

def four(ax_three):

    start = 1970
    end = 2020
    yearTotal = end - start + 1

    # year family range
    reserve = load_variavle(r'chugao\fish_220114_re\correlation_median_point\reserve_family.txt')
    countByYear = load_variavle('chugao\\fish_220114_re\\countByYear_1degree_1970_2020_2846_1.txt')
    record = np.array(countByYear)

    percentUp = [[[] for ran in range(3)] for year in range(yearTotal - 3)]
    percentDown = [[[] for ran in range(3)] for year in range(yearTotal - 3)]

    for year in range(yearTotal - 3):
        total = np.sum([record[year: year + 4, old, 90:] for new, old in reserve])
        print(year,total)
        for family in range(2846):
            if family in [old for new,old in reserve]:
                ranges1 = np.array([np.sum(record[year: year + 3, family, 90:120]), np.sum(record[year: year + 3, family, 120:150]), np.sum(record[year: year + 3, family, 150:])])
                ranges2 = np.array([np.sum(record[year + 1: year + 4, family, 90:120]), np.sum(record[year + 1: year + 4, family, 120:150]), np.sum(record[year + 1: year + 4, family, 150:])])
                for ran in range(3):
                    if np.sum(record[year + 1: year + 4, family, 90 + ran * 30:120 + ran * 30]) == np.max(ranges2) and not np.sum(record[year: year + 3, family, 90 + ran * 30:120 + ran * 30]) == np.max(ranges1):
                        percentUp[year][ran].append(np.sum(record[year: year + 4, family, 90:]) / total)
                    if np.sum(record[year: year + 3, family, 90 + ran * 30:120 + ran * 30]) == np.max(ranges1) and not np.sum(record[year + 1: year + 4, family, 90 + ran * 30:120 + ran * 30]) == np.max(ranges2):
                        percentDown[year][ran].append(np.sum(record[year: year + 4, family, 90:]) / total)

    aveFamilyUp = np.array([[0.0 for ran in range(3)] for year in range(yearTotal - 3)])
    aveFamilyDown = np.array([[0.0 for ran in range(3)] for year in range(yearTotal - 3)])
    aveFamilyUpNum = np.array([[0.0 for ran in range(3)] for year in range(yearTotal - 3)])
    aveFamilyDownNum = np.array([[0.0 for ran in range(3)] for year in range(yearTotal - 3)])

    for year in range(yearTotal - 3):
        for ran in range(3):
            aveFamilyUp[year, ran] = np.mean(percentUp[year][ran])
            aveFamilyDown[year, ran] = np.mean(percentDown[year][ran])
            aveFamilyUpNum[year, ran] = np.sum(percentUp[year][ran])
            aveFamilyDownNum[year, ran] = np.sum(percentDown[year][ran])

    # sliding windows
    aveUp = np.array([[0.0 for ran in range(3)] for year in range(yearTotal - 12)])
    aveDown = np.array([[0.0 for ran in range(3)] for year in range(yearTotal - 12)])
    for year in range(yearTotal - 12):
        for ran in range(3):
            aveUp[year, ran] = np.mean(aveFamilyUpNum[year: year + 10, ran][aveFamilyUpNum[year: year + 10, ran] > 0])
            aveDown[year, ran] = np.mean(aveFamilyDownNum[year: year + 10, ran][aveFamilyUpNum[year: year + 10, ran] > 0])


    colors = [(236/255,95/255,116/255,1),(255/255,111/255,105/255,1),(160/255,64/255,160/255,1),(205/255,62/255,205/255,1),(46/255,117/255,182/255,1),(52/255,152/255,219/255,1)]
    reg = ['0°N~30°N','30°N~60°N','60°N~90°N']
    for ran in range(3):
        ax_three.bar(np.arange(1975,2026-12,1) + 0.3 * ran, aveUp[:, ran], color = colors[ran * 2],width = 0.3, label = 'Shifting into {}'.format(reg[ran]))
        ax_three.bar(np.arange(1975,2026-12,1) + 0.3 * ran, -aveDown[:, ran], color = colors[ran * 2 + 1], width = 0.3, label = 'Shifting out of {}'.format(reg[ran]))
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
    font3 = {
         'weight': 'bold',
		 'style':'normal',
         'size': 10,
         }
    font4 = {
         'weight': 'bold',
		 'style':'normal',
         'size': 30,
         }
    font5 = {
		 'style':'normal',
         'size': 10,
         }
         
    # plt.ylabel('Average family',fontsize = 20)
    ax_three.set_ylabel('Percent Species Shifting',font1,labelpad=15)
    ax_three.set_xticks([1975,1984,1993,2002,2011])
    ax_three.set_yticks([-0.03,-0.02,-0.01,0,0.01,0.02,0.03])

    ax_three.set_xticklabels(['1970-1982','1979-1991','1988-2000','1997-2009','2006-2018'],font1)
    ax_three.set_yticklabels(['3%','2%','1%','0%','1%','2%','3%'],font1)
    ax_three.set_xlabel('Year',font1)
    ax_three.set_xlim([1972,2017])
    ax_three.set_ylim([-0.04,0.04])
    ax_three.legend(prop = font2,ncol = 3,loc = 3,fontsize = 12, columnspacing = 0.5)
    set_axis(ax_three)
    ax_three.text(1972.5, -0.04 + (0.04 + 0.04) * 0.97, 'D', verticalalignment="top",
                horizontalalignment="left", fontdict=font4)
    # fig = plt.gcf()
    # fig.set_size_inches(20, 10)

    '''
    LOWESS regression
    '''
    def get_p_value(arrA, arrB):
        a = np.array(arrA)
        b = np.array(arrB)
        t, p = stats.ttest_ind(a,b)
        return 1 - p

    left,bottom,width,height = [0.68,0.65,0.3,0.3]
    # ax2 = fig.add_axes([left,bottom,width,height])
    ax2 = ax_three.inset_axes((left,bottom,width,height))

    left,bottom,width,height = [0.68,0.1,0.3,0.3]
    # ax3 = fig.add_axes([left,bottom,width,height])
    ax3 = ax_three.inset_axes((left,bottom,width,height))
    for ran in range(3):
        for half in range(2):
            smoother = LowessSmoother(smooth_fraction=0.4, iterations=1)
            if half == 0:
                initial = aveUp[:, ran]     
            else:
                initial = aveDown[:, ran]
            smoother.smooth(initial)
            y_pred = smoother.smooth_data[0]
            if half == 0:
                ax2.plot(np.arange(1975,2026-12,1), y_pred, color = colors[ran * 2], label ='$\\it{p}$ = ' + '%.4f' % get_p_value(initial, y_pred))
            else:
                ax3.plot(np.arange(1975,2026-12,1), y_pred, '--', color = colors[ran * 2 + 1], label = '$\\it{p}$ = ' +'%.4f' % get_p_value(initial, y_pred))
    
    leg2 = ax2.legend(prop = font5, loc=1, fontsize = 10,labelspacing = 0.2)
    leg3 = ax3.legend(prop = font5, loc=1, fontsize = 10,labelspacing = 0.2)

    # plt.setp(leg2.get_texts(), fontweight='bold')
    # plt.setp(leg3.get_texts(), fontweight='bold')

    ax2.set_xticks([1980,1995,2010])
    ax2.set_yticks([0,0.01,0.02])
    ax2.set_xticklabels(['1975-1987','1990-2002','2005-2017'],font3)
    ax2.set_yticklabels(['0%','1%','2%'],font3)
    ax2.set_ylim([-0.005,0.025])
    ax2.grid(False)
    ax3.set_xticks([1980,1995,2010])
    ax3.set_yticks([0,0.01,0.02])
    ax3.set_xticklabels(['1975-1987','1990-2002','2005-2017'],font3)
    ax3.set_yticklabels(['0%','1%','2%'],font3)
    ax3.set_ylim([-0.005,0.025])
    ax3.grid(False)

    ax2.spines['top'].set_color('black')
    ax2.spines['bottom'].set_color('black')
    ax2.spines['left'].set_color('black')
    ax2.spines['right'].set_color('black')
    ax3.spines['top'].set_color('black')
    ax3.spines['bottom'].set_color('black')
    ax3.spines['left'].set_color('black')
    ax3.spines['right'].set_color('black')

    ax2.text(1975,-0.003,'D$_1$ Fitting trends (shifting into regions)',verticalalignment="bottom",horizontalalignment="left",fontdict=font2)
    ax3.text(1975,-0.003,'D$_2$ Fitting trends (shifting out of regions)',verticalalignment="bottom",horizontalalignment="left",fontdict=font2)
    for tickline in ax2.xaxis.get_ticklines():
        tickline.set_visible(True)
    for tickline in ax2.yaxis.get_ticklines():
        tickline.set_visible(True)
    for tickline in ax3.xaxis.get_ticklines():
        tickline.set_visible(True)
    for tickline in ax3.yaxis.get_ticklines():
        tickline.set_visible(True)