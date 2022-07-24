import matplotlib.pyplot as plt
import matplotlib.gridspec as mg
import numpy as np
import fig3A
import fig3C
import fig3B

'''
    0  1  2  3  4  5  6  7  8  9  10  11  12
0   Ⅰ Ⅰ Ⅰ Ⅰ Ⅰ Ⅰ Ⅰ  Ⅱ Ⅱ Ⅱ  Ⅳ Ⅳ Ⅳ
1   Ⅰ Ⅰ Ⅰ Ⅰ Ⅰ Ⅰ Ⅰ  Ⅲ Ⅲ Ⅲ  Ⅲ Ⅲ Ⅲ

'''
# ax_one = plt.subplot2grid((2,13),(0,0),rowspan=2,colspan=7)
# ax_one.patch.set_facecolor('None')
# ax_subone = ax_one.twiny()
# ax_two = plt.subplot2grid((2,13),(0,7),rowspan=1,colspan=3)
# fig3A.one_two(ax_one, ax_subone, ax_two)
# ax_three = plt.subplot2grid((2,13),(1,7),rowspan=1,colspan=6)
# fig3C.three(ax_three)
# ax_four = plt.subplot2grid((2,13),(0,10),rowspan=1,colspan=3)
# fig3B.four(ax_four)
# plt.subplots_adjust(wspace=1.2,hspace=0.16)
# fig = plt.gcf()
# fig.set_size_inches(28, 14)
# # fig.savefig(r'chugao\fish_220114_re\figure2\fig2.png',dpi=1000)
# plt.show()

'''
    0  1  2  3  4  5 
0   Ⅱ Ⅱ Ⅱ  Ⅳ Ⅳ Ⅳ
1   Ⅲ Ⅲ Ⅲ  Ⅲ Ⅲ Ⅲ

'''
plt.figure(1)
ax_one = plt.subplot(111)
ax_one.patch.set_facecolor('None')
ax_subone = ax_one.twiny()
plt.figure(2)
ax_two = plt.subplot(111)
fig3A.one_two(ax_one, ax_subone, ax_two)
plt.figure(1)
fig1 = plt.gcf()
fig1.set_size_inches(18, 13)
plt.figure(3)
ax_two = plt.subplot2grid((2,6),(0,0),rowspan=1,colspan=3)
fig3A.one_two(ax_one, ax_subone, ax_two)
ax_three = plt.subplot2grid((2,6),(1,0),rowspan=1,colspan=6)
fig3C.three(ax_three)
ax_four = plt.subplot2grid((2,6),(0,3),rowspan=1,colspan=3)
fig3B.four(ax_four)
plt.subplots_adjust(wspace=1.2,hspace=0.16)
fig = plt.gcf()
fig.set_size_inches(20, 15)
# fig1.savefig(r'chugao\fish_220114_re\SM\figS4.png',dpi=1000)
# fig.savefig(r'chugao\fish_220114_re\figure2\fig2_new.png',dpi=1000)
plt.show()