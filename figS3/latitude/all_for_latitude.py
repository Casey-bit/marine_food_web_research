import early
import late
import matplotlib.pyplot as plt
from axes_frame import set_axis

ax1 = plt.subplot(121)
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



ax2 = plt.subplot(122)
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

plt.show()
