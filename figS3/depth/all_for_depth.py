import early_vertical
import late_vertical
import matplotlib.pyplot as plt
from axes_frame import set_axis

ax1 = plt.subplot(121)
early_vertical.early_shift(ax1)
ax1.set_xlim([-5,43])
ax1.set_ylim([-100,3000])
# ax1.set_yscale('log',base=5)

set_axis(ax1)
ax1.set_xticks([0,10,20,30,40])
ax1.set_xticklabels(['1970-1980','1980-1990','1990-2000','2000-2010','2010-2020'])
# ax1.set_yticks([0, 50, 100, 150, 200])
# ax1.set_yticklabels(['0','50','100','150','200'])
# ax1.set_yticks([10, 20, 30, 40, 60, 70, 80, 90, 110, 120, 130, 140, 160, 170, 180, 190, 210], minor=True)
ax1.text(0, 2800, 'Range determination according to\nthe earlier 20 years (1970 - 1990)', fontsize=12, weight='bold')
font2 = {
    'weight': 'bold',
    'style': 'normal',
    'size': 20,
}
ax1.set_ylabel('Depth (m)', font2)


ax2 = plt.subplot(122)
late_vertical.late_shift(ax2)

ax2.set_xlim([-5,43])
ax2.set_ylim([-100,3000])
# ax2.set_yscale('log',base=5)

set_axis(ax2)
ax2.set_xticks([0,10,20,30,40])
ax2.set_xticklabels(['1970-1980','1980-1990','1990-2000','2000-2010','2010-2020'])
# ax2.set_yticks([0, 50, 100, 150, 200])
# ax2.set_yticklabels(['0','50','100','150','200'])
# ax2.set_yticks([10, 20, 30, 40, 60, 70, 80, 90, 110, 120, 130, 140, 160, 170, 180, 190, 210], minor=True)
ax2.text(0, 2800, 'Range determination according to\nthe later 20 years (2000 - 2020)', fontsize=12, weight='bold')

plt.show()
