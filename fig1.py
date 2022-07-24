from statistics import median
import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import pywt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle

from torch import arange
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

fifty = load_variavle(r'chugao\fish_220114_re\correlation_median_point\medianPoint.txt')
reserve = load_variavle(r'chugao\fish_220114_re\correlation_median_point\reserve.txt')
# print(reserve)

# # 小波滤噪
# def wavelet_denoising(data):
#     # 小波函数取db4
#     db4 = pywt.Wavelet('db4')
#     # 分解
#     coeffs = pywt.wavedec(data, db4)
#     # 高频系数置零
#     coeffs[len(coeffs)-1] *= 0
#     coeffs[len(coeffs)-2] *= 0
#     # 重构
#     meta = pywt.waverec(coeffs, db4)
#     return meta

migration = [[[[] for after in range(9)] for init in range(9)] for r in range(4)] # 迁移类别,e.g. 0-10 -> 20-30; 50-60 -> 10-20
mig_onerange_down = [[[] for after in range(9)] for r in range(4)]

for i in range(fifty.shape[0]):
    if i in [after for init, after in reserve]:
        result = fifty[i]
        lat = np.array([0.0 for c in range(5)])
        lat[0] = np.mean(result[0:10][result[0:10] > 0])
        lat[1] = np.mean(result[10:20][result[10:20] > 0])
        lat[2] = np.mean(result[20:30][result[20:30] > 0])
        lat[3] = np.mean(result[30:40][result[30:40] > 0])
        lat[4] = np.mean(result[40:][result[40:] > 0])
        for c in range(1,5):
            if np.isnan(lat[c]):
                lat[c] = lat[c - 1]
        for c in range(4):
            if int(lat[c] / 10) == int(lat[c + 1] / 10) and lat[c] > lat[c + 1]:
                mig_onerange_down[c][int(lat[c] / 10)].append(i)
            else:
                migration[c][int(lat[c]/10)][int(lat[c + 1]/10)].append(i)

# Source和Target是Plotly将要连接的节点的索引列表，Value是定义这些连接宽度的数字列表。

# data
label = ["0°N~10°N", "10°N~20°N", "20°N~30°N", "30°N~40°N", "40°N~50°N", "50°N~60°N", "60°N~70°N", "70°N~80°N", "80°N~90°N","0°N~10°N", "10°N~20°N", "20°N~30°N", "30°N~40°N", "40°N~50°N", "50°N~60°N", "60°N~70°N", "70°N~80°N", "80°N~90°N","0°N~10°N", "10°N~20°N", "20°N~30°N", "30°N~40°N", "40°N~50°N", "50°N~60°N", "60°N~70°N", "70°N~80°N", "80°N~90°N","0°N~10°N", "10°N~20°N", "20°N~30°N", "30°N~40°N", "40°N~50°N", "50°N~60°N", "60°N~70°N", "70°N~80°N", "80°N~90°N","0°N~10°N", "10°N~20°N", "20°N~30°N", "30°N~40°N", "40°N~50°N", "50°N~60°N", "60°N~70°N", "70°N~80°N", "80°N~90°N"]
source = []
target = []
value = []
lcolor = []
for r in range(4):
    for i in range(9):
        source += [i + 9 * r for c in range(9)]
        target += list(np.arange(9 + 9 * r,18 + 9 * r,1))
        for j in range(9):
            # if 0 < len(migration[r][i][j]) < 10:
            #     value.append(10)
            # else:
            value.append(len(migration[r][i][j]))
            if i <= j:
                lcolor.append('rgba(255,151,79,{})'.format(0.5+0.1*r))
            else:
                lcolor.append('rgba(52,152,219,{})'.format(0.5+0.1*r))
        source.append(i + 9 * r)
        target.append(9 + i + 9 * r)
        value.append(len(mig_onerange_down[r][i]))
        lcolor.append('rgba(52,152,219,{})'.format(0.5+0.1*r))

x = [0.1 for i in range(7)]
y = [1,0.88,0.76,0.64,0.52,0.38,0.24]
color = ['rgba(238,103,76,0.5)' for i in range(9)]

x += [0.3 for i in range(7)]
y += [1,0.88,0.76,0.64,0.52,0.38,0.24]
color += ['rgba(238,103,76,0.6)' for i in range(9)]

x += [0.5 for i in range(8)]
y += [1,0.88,0.76,0.64,0.52,0.38,0.24,0.12]
color += ['rgba(238,103,76,0.7)' for i in range(9)]

x += [0.7 for i in range(9)]
y += [1,0.88,0.76,0.64,0.52,0.38,0.24,0.12,0.001]
color += ['rgba(238,103,76,0.8)' for i in range(9)]

x += [0.9 for i in range(9)]
y += [1,0.88,0.76,0.64,0.52,0.38,0.24,0.12,0.001]
color += ['rgba(238,103,76,0.9)' for i in range(9)]

link = dict(source = source, target = target, value = value)
node = dict(label = label, pad=50, thickness=20)

fig = make_subplots(
    rows=1, cols=2,  # 5*2的图形
)
fig.add_trace(
    go.Scatter(x=[1, 2, 3], y=[5, 10, 15]),
    row=1, col=1  # 第一行第一列
)

data = go.Sankey(link = link, node=node)# plot
fig = go.Figure(data)

fig.update_traces(node_x=x, selector=dict(type='sankey'))
fig.update_traces(node_y=y, selector=dict(type='sankey'))
fig.update_traces(node_color=color, selector=dict(type='sankey'))
fig.update_traces(link_color=lcolor, selector=dict(type='sankey'))
fig.update_traces(textfont_size=15, selector=dict(type='sankey'))
fig.update_traces(node_line_width=2, selector=dict(type='sankey'))
fig.update_traces(node_line_color='rgba(238,103,76,1)', selector=dict(type='sankey'))
fig.update_layout(
    annotations=[
        dict(
            x=0.95, y=-0.1,
            text='',
            showarrow=True, arrowhead=4, arrowwidth=12, arrowsize=0.7,
            axref='x domain', ax=-1200,
            ayref='y domain', ay=0,
            font={'size': 16, 'color': '#ffffff', 'family': 'Courier New, monospace'},
        ),
        dict(
            x=0.1, y=-0.1,
            text='1970s',
            showarrow=True, arrowhead=4, arrowwidth=12, arrowsize=0.7,
            axref='x domain', ax=-0,
            ayref='y domain', ay=0,
            font={'size': 16, 'color': '#ffffff', 'family': 'Courier New, monospace'},
            bgcolor='rgba(184,112,112,0.5)', borderpad=4, bordercolor='#c7c7c7', borderwidth=2,
            standoff=5, 
        ),
        dict(
            x=0.3, y=-0.1,
            text='1980s',
            showarrow=True, arrowhead=4, arrowwidth=12, arrowsize=0.7,
            axref='x domain', ax=-0,
            ayref='y domain', ay=0,
            font={'size': 16, 'color': '#ffffff', 'family': 'Courier New, monospace'},
            bgcolor='rgba(184,112,112,0.6)', borderpad=4, bordercolor='#c7c7c7', borderwidth=2,
            standoff=5, 
        ),
        dict(
            x=0.5, y=-0.1,
            text='1990s',
            showarrow=True, arrowhead=4, arrowwidth=12, arrowsize=0.7,
            axref='x domain', ax=-0,
            ayref='y domain', ay=0,
            font={'size': 16, 'color': '#ffffff', 'family': 'Courier New, monospace'},
            bgcolor='rgba(184,112,112,0.7)', borderpad=4, bordercolor='#c7c7c7', borderwidth=2,
            standoff=5, 
        ),
        dict(
            x=0.7, y=-0.1,
            text='2000s',
            showarrow=True, arrowhead=4, arrowwidth=12, arrowsize=0.7,
            axref='x domain', ax=-0,
            ayref='y domain', ay=0,
            font={'size': 16, 'color': '#ffffff', 'family': 'Courier New, monospace'},
            bgcolor='rgba(184,112,112,0.8)', borderpad=4, bordercolor='#c7c7c7', borderwidth=2,
            standoff=5, 
        ),
        dict(
            x=0.9, y=-0.1,
            text='2010s',
            showarrow=True, arrowhead=4, arrowwidth=12, arrowsize=0.7,
            axref='x domain', ax=-0,
            ayref='y domain', ay=0,
            font={'size': 16, 'color': '#ffffff', 'family': 'Courier New, monospace'},
            bgcolor='rgba(184,112,112,0.9)', borderpad=4, bordercolor='#c7c7c7', borderwidth=2,
            standoff=5, 
        )
    ]
)

                  
fig.show()