import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.transforms import TransformedBbox, Bbox
from matplotlib.image import BboxImage
from matplotlib.legend_handler import HandlerBase,HandlerTuple
import os

font1 = {
    'weight': 'bold',
    'style': 'normal',
    'size': 15,
}
font2 = {
    'weight': 'bold',
    'style': 'normal',
    'size': 12,
}
font3 = {
    'weight': 'bold',
    'style': 'normal',
    'size': 20,
}


class ImageHandler(HandlerBase):
    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        # 缩放图像
        sx, sy = self.image_stretch
        # 创建边框用于放置图像
        bb = Bbox.from_bounds(xdescent - sx, ydescent -
                              sy, width + sx, height + sy)
        tbb = TransformedBbox(bb, trans)
        image = BboxImage(tbb)
        image.set_data(self.image_data)
        # 如果绑定的可视对象不在默认映射范围内，需要注释掉该语句
        self.update_prop(image, orig_handle, legend)
        return [image]

    def set_image(self, image_path, image_stretch=(0, 0)):
        if os.path.exists(image_path):
            self.image_data = plt.imread(image_path)
        self.image_stretch = image_stretch


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
    for tickline in ax.yaxis.get_minorticklines():
        tickline.set_visible(True)
    ax.tick_params(which="major",
                   length=15, width=2.0,
                   colors="black",
                   direction='in',
                   tick2On=False,
                   label2On=False)
    ax.tick_params(which="minor",
                   length=5, width=1.0,
                   colors="black",
                   direction='in',
                   tick2On=False,
                   label2On=False)

    ax.set_ylim([0, 6])
    ax.set_xticklabels(ax.get_xticklabels(), font1)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels([1, 2, 3, 4, 5], font1)
    ax.set_ylabel("Family Trophic Level", font1)
    ax.set_xlabel("Family Shift Category", font1)


ax = plt.subplot()
df = pd.read_csv(r'vertex0.9_node.csv')
total = [df[df['mode'] == 0], df[df['mode'] == 1],
         df[df['mode'] == 2], df[df['mode'] == 3]]

for idx in range(4):
    level = total[idx]['level'].values.tolist()
    # num = total[idx]['family_num'].values.tolist()
    level_prop = [len([i for i in level if i == 1]) / len(level),
                  len([i for i in level if i == 2]) / len(level),
                  len([i for i in level if i == 3]) / len(level),
                  len([i for i in level if i == 4]) / len(level),
                  len([i for i in level if i == 5]) / len(level)]
    # level_prop = [0,0,0,0,0]
    # for lev, nu in zip(level, num):
    #     level_prop[lev - 1] += nu
    # allnum = np.sum(level_prop)
    # print(allnum, level_prop)
    # level_prop = [i / allnum for i in level_prop]

    level_prop = [i * 1.5 for i in level_prop]

    data_left = [idx + 1 - i/2 for i in level_prop]
    data_right = [idx + 1 + i/2 for i in level_prop]
    color_list = [(140/255, 185/255, 0/255), (20/255, 178/255, 255/255), (255/255, 194/255, 102/255),
                  (255/255, 112/255, 69/255), (255/255, 26/255, 128/255)]  # 柱子颜色

    axes = []
    for l in range(5):
        axes.append(ax.barh(y=l + 1, width=level_prop[l], left=data_left[l],
                    color=color_list[l], height=0.9, label='{}'.format(l + 1)))  # 柱宽设置为0.7
    if idx == 0:
        s = [plt.scatter(-1, -1) for i in range(5)]
        custom_handler1 = ImageHandler()
        custom_handler1.set_image(r"E:\paper\label\one.png",image_stretch=(2,2))

        custom_handler2 = ImageHandler()
        custom_handler2.set_image(r"E:\paper\label\two.png",image_stretch=(4,4))

        custom_handler3 = ImageHandler()
        custom_handler3.set_image(r"E:\paper\label\three.png",image_stretch=(6,6))

        custom_handler4 = ImageHandler()
        custom_handler4.set_image(r"E:\paper\label\four.png",image_stretch=(8,8))

        custom_handler5 = ImageHandler()
        custom_handler5.set_image(r"E:\paper\label\five.png",image_stretch=(10,10))

        handles, labels = ax.get_legend_handles_labels()
        hand = zip(s, axes)
        lg = plt.legend(
            hand,
            labels[:5],
            handler_map={tuple: HandlerTuple(
                ndivide=None), s[0]: custom_handler1, s[1]: custom_handler2, s[2]: custom_handler3, s[3]: custom_handler4, s[4]: custom_handler5},
            prop = font1,
            bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
            mode="expand", borderaxespad=0.,ncol = 5
        )

        # leng = plt.legend(prop = font2,ncol = 5,loc = 1, columnspacing = 0.5)
        lg.set_title(title='Trophic level', prop = font1)

    polygons = []
    for i in range(5):
        # 阶段
        ax.text(
            idx + 1,
            i + 1,
            "%.2f" % (level_prop[i] / 1.5 * 100) + '%', 
            color='black', alpha=0.8, ha="center", va='center',
            fontdict=font2)

        if i < 4:
            polygons.append(Polygon(xy=np.array([(data_left[i+1], i + 2 - 0.45),
                                                (data_right[i+1],
                                                 i + 2 - 0.45),
                                                (data_right[i], i + 1 + 0.45),
                                                (data_left[i], i + 1 + 0.45)])))

    ax.add_collection(PatchCollection(polygons,
                                      facecolor='#e2b0b0',
                                      alpha=0.8))


set_axis(ax)
ax.set_xlim([0.5, 4.5])
ax.set_ylim([0.3, 5.7])
ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels(['Total', 'Northward', 'Southward', 'Mixed'], font1)
ax.set_yticks([1, 2, 3, 4, 5])
ax.set_yticklabels([1, 2, 3, 4, 5], font1)
ax.text(0.6, 0.3 + 5.4 * 0.95, 'B', font3)
fig = plt.gcf()
fig.set_size_inches(8, 8)
fig.savefig(r'chugao\fish_220114_re\figure2\fig2B.pdf', dpi=500)

plt.show()
