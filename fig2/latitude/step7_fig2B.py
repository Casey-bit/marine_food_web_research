import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Polygon
from matplotlib.figure import Figure
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.collections import PatchCollection
from matplotlib.transforms import TransformedBbox, Bbox
from matplotlib.image import BboxImage
from matplotlib.legend_handler import HandlerBase
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
import os
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

        sx, sy = self.image_stretch

        bb = Bbox.from_bounds(xdescent - sx, ydescent -
                              sy, width + sx, height + sy)
        tbb = TransformedBbox(bb, trans)
        image = BboxImage(tbb)
        image.set_data(self.image_data)

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


df = pd.read_csv(r'E:\gephi\data\vertex0.9_cluster_220125.csv')
node = [] 
df['source'] = df['source'].astype(int)
df['target'] = df['target'].astype(int)

for elem in df['source']:
    if not elem in node:
        node.append(elem)

for elem in df['target']:
    if not elem in node:
        node.append(elem)

reserve = load_variavle(r'fig2\latitude\reserve.txt')
median = load_variavle(r'fig2\latitude\median_denoising.txt')

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

node_df = pd.read_csv(r'E:\gephi\data\vertex0.9_node_220125.csv')
node_df = node_df[['id', 'level']]

mk_test_res = pd.merge(mk_test_res, node_df, left_on=['family'], right_on=['id'])
mk_test_res_2 = mk_test_res.copy()
mk_test_res_2['describe'] = 'Total'
mk_test_res_2['mode'] = 0

mk_test_res = mk_test_res.append(mk_test_res_2, ignore_index = True)


ax = plt.subplot()
df = mk_test_res
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
                  (255/255, 112/255, 69/255), (255/255, 26/255, 128/255)]

    axes = []
    for l in range(5):
        axes.append(ax.barh(y=l + 1, width=level_prop[l], left=data_left[l],
                    color=color_list[l], height=0.9, label='{}'.format(l + 1))) 
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

    # if idx == 1:
    #     arr_img = plt.imread(r"E:\paper\label\two.png")
    #     im = OffsetImage(arr_img, zoom=0.3)
    #     ab = AnnotationBbox(im, (0.2, 2))
    #     ax.add_artist(ab)

    # if idx == 2:
    #     arr_img = plt.imread(r"E:\paper\label\three1.png")
    #     im = OffsetImage(arr_img, zoom=0.3)
    #     ab = AnnotationBbox(im, (0.2, 3))
    #     ax.add_artist(ab)

    # if idx == 3:
    #     arr_img = plt.imread(r"E:\paper\label\four1.png")
    #     im = OffsetImage(arr_img, zoom=0.3)
    #     ab = AnnotationBbox(im, (0.2, 4))
    #     ax.add_artist(ab)
    #     arr_img = plt.imread(r"E:\paper\label\five2.png")
    #     im = OffsetImage(arr_img, zoom=0.3)
    #     ab = AnnotationBbox(im, (0.2, 5))
    #     ax.add_artist(ab)

    polygons = []
    for i in range(5):
        ax.text(
            idx + 1,  # location
            i + 1,  # height
            "%.2f" % (level_prop[i] / 1.5 * 100) + '%',  # text
            color='black', alpha=0.8, ha="center", va='center',
            fontdict=font2)

        # ax.text(
        #   data2[0] / 2 ,
        #   i,
        #   str(data[::-1][i]) +'(' +str(round(data[::-1][i] / data[0] * 100, 1)) + '%)',
        #   color='black', alpha=0.8, size=18, ha="center")

        if i < 4:
            # ax.text(
            #   data2[0] / 2 ,
            #   4.4 - i,
            #   str(round(data[i+1] / data[i], 3) * 100) + '%',
            #   color='black', alpha=0.8, size=16, ha="center")

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
fig.savefig(r'E:\paper\202301\fig\fig2\latitudeB_lati.pdf', dpi=500)

plt.show()
