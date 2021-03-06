# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np
import sklearn.cluster as sc
import matplotlib.pyplot as mp
x = []
with open('../../data/multiple3.txt', 'r') as f:
    for line in f.readlines():
        data = [float(substr) for substr
                in line.split(',')]
        x.append(data)
x = np.array(x)
# 量化带宽，决定每次调整概率密度函数的步进量
bw = sc.estimate_bandwidth(x, n_samples=len(x),
                           quantile=0.1)
# 均值漂移聚类器
model = sc.MeanShift(bandwidth=bw, bin_seeding=True)
model.fit(x)
centers = model.cluster_centers_
l, r, h = x[:, 0].min() - 1, x[:, 0].max() + 1, 0.005
b, t, v = x[:, 1].min() - 1, x[:, 1].max() + 1, 0.005
grid_x = np.meshgrid(np.arange(l, r, h),
                     np.arange(b, t, v))
flat_x = np.c_[grid_x[0].ravel(), grid_x[1].ravel()]
flat_y = model.predict(flat_x)
grid_y = flat_y.reshape(grid_x[0].shape)
pred_y = model.predict(x)
mp.figure('Mean Shift Cluster', facecolor='lightgray')
mp.title('Mean Shift Cluster', fontsize=20)
mp.xlabel('x', fontsize=14)
mp.ylabel('y', fontsize=14)
mp.tick_params(labelsize=10)
mp.pcolormesh(grid_x[0], grid_x[1], grid_y,
              cmap='gray')
mp.scatter(x[:, 0], x[:, 1], c=pred_y, cmap='brg',
           s=80)
mp.scatter(centers[:, 0], centers[:, 1], marker='+',
           c='gold', s=1000, linewidth=1)
mp.show()
