import early
import late
import matplotlib.pyplot as plt
from axes_frame import set_axis

font2 = {
    'weight': 'bold',
    'style': 'normal',
    'size': 20,
}

ax1 = plt.subplot(221)
early.early_shift(ax1)
ax1.set_xlim([-5,43])
ax1.set_ylim([0,90])
ax1.annotate(s='', xy=(-3, 2), xytext=(-3, 28), arrowprops=dict(
    color='black', arrowstyle='<->,head_width=0.5,head_length = 1.5', linewidth=3))
ax1.annotate(s='', xy=(-3, 32), xytext=(-3, 58), arrowprops=dict(
    color='black', arrowstyle='<->,head_width=0.5,head_length = 1.5', linewidth=3))
ax1.annotate(s='', xy=(-3, 62), xytext=(-3, 88), arrowprops=dict(
    color='black', arrowstyle='<->,head_width=0.5,head_length = 1.5', linewidth=3))
ax1.text(-1.7, 11, 'Region 1', rotation=90,
            fontsize=12, weight='bold')
ax1.text(-1.7, 41, 'Region 2', rotation=90,
            fontsize=12, weight='bold')
ax1.text(-1.7, 71, 'Region 3', rotation=90,
            fontsize=12, weight='bold')
set_axis(ax1)
ax1.set_xticks([0,10,20,30,40])
ax1.set_xticklabels(['1970-1980','1980-1990','1990-2000','2000-2010','2010-2020'])
ax1.set_yticks([0, 30, 60, 90])
ax1.set_yticklabels(['0','30','60','90'])
ax1.set_yticks([10, 20, 40, 50, 70, 80], minor=True)
ax1.text(2, 85, 'Range determination according to the earlier 20 years (1970 - 1990)', fontsize=12, weight='bold')
ax1.text(-1, 85, 'A', fontsize=30, weight='bold')
ax1.set_ylabel('Latitude (°N)', font2)


ax2 = plt.subplot(222)
late.late_shift(ax2)

ax2.annotate(s='', xy=(-3, 2), xytext=(-3, 28), arrowprops=dict(
    color='black', arrowstyle='<->,head_width=0.5,head_length = 1.5', linewidth=3))
ax2.annotate(s='', xy=(-3, 32), xytext=(-3, 58), arrowprops=dict(
    color='black', arrowstyle='<->,head_width=0.5,head_length = 1.5', linewidth=3))
ax2.annotate(s='', xy=(-3, 62), xytext=(-3, 88), arrowprops=dict(
    color='black', arrowstyle='<->,head_width=0.5,head_length = 1.5', linewidth=3))
ax2.text(-1.7, 11, 'Region 1', rotation=90,
            fontsize=12, weight='bold')
ax2.text(-1.7, 41, 'Region 2', rotation=90,
            fontsize=12, weight='bold')
ax2.text(-1.7, 71, 'Region 3', rotation=90,
            fontsize=12, weight='bold')
ax2.set_xlim([-5,43])
ax2.set_ylim([0,90])

set_axis(ax2)
ax2.set_xticks([0,10,20,30,40])
ax2.set_xticklabels(['1970-1980','1980-1990','1990-2000','2000-2010','2010-2020'])
ax2.set_yticks([0, 30, 60, 90])
ax2.set_yticklabels(['0','30','60','90'])
ax2.set_yticks([10, 20, 40, 50, 70, 80], minor=True)
ax2.text(2, 85, 'Range determination according to the later 20 years (2000 - 2020)', fontsize=12, weight='bold')
ax2.text(-1, 85, 'B', fontsize=30, weight='bold')


import early_vertical
import late_vertical

ax1 = plt.subplot(223)
early_vertical.early_shift(ax1)
ax1.set_xlim([-5,43])
ax1.set_ylim([-100,3000])
# ax1.set_yscale('log',base=5)

set_axis(ax1)
ax1.set_xticks([0,10,20,30,40])
ax1.set_xticklabels(['1970-1980','1980-1990','1990-2000','2000-2010','2010-2020'])
ax1.set_yticks([0, 500, 1000, 1500, 2000,2500,3000])
ax1.set_yticklabels([0, 500, 1000, 1500, 2000,2500,3000])
# ax1.set_yticks([10, 20, 30, 40, 60, 70, 80, 90, 110, 120, 130, 140, 160, 170, 180, 190, 210], minor=True)
ax1.text(0, 2800, 'Range determination according to\nthe earlier 20 years (1970 - 1990)', fontsize=12, weight='bold')
ax1.text(-3, 2800, 'C', fontsize=30, weight='bold')

ax1.set_xlabel('Year', font2)
ax1.set_ylabel('Depth (m)', font2)


ax2 = plt.subplot(224)
late_vertical.late_shift(ax2)

ax2.set_xlim([-5,43])
ax2.set_ylim([-100,3000])
# ax2.set_yscale('log',base=5)

set_axis(ax2)
ax2.set_xticks([0,10,20,30,40])
ax2.set_xticklabels(['1970-1980','1980-1990','1990-2000','2000-2010','2010-2020'])
ax2.set_yticks([0, 500, 1000, 1500, 2000,2500,3000])
ax2.set_yticklabels([0, 500, 1000, 1500, 2000,2500,3000])
# ax2.set_yticks([10, 20, 30, 40, 60, 70, 80, 90, 110, 120, 130, 140, 160, 170, 180, 190, 210], minor=True)
ax2.text(0, 2800, 'Range determination according to\nthe later 20 years (2000 - 2020)', fontsize=12, weight='bold')
ax2.text(-3, 2800, 'D', fontsize=30, weight='bold')
ax2.set_xlabel('Year', font2)

plt.subplots_adjust(wspace=0.1,hspace=0.15)
fig = plt.gcf()
fig.set_size_inches(22, 22)
fig.savefig(r'cluster\figS2.jpg', dpi=150)
plt.show()
