# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np
import sklearn.cluster as sc
import matplotlib.pyplot as mp
n_samples = 500
t = 2.5 * np.pi * (1 + 2 * np.random.rand(
    n_samples, 1))
x = 0.05 * t * np.cos(t)
y = 0.05 * t * np.sin(t)
n = 0.05 * np.random.rand(n_samples, 2)
x = np.hstack((x, y)) + n
mp.figure('Agglomerative Cluster',
          facecolor='lightgray')
mp.title('Agglomerative Cluster', fontsize=20)
mp.xlabel('x', fontsize=14)
mp.ylabel('y', fontsize=14)
mp.tick_params(labelsize=10)
mp.axis('equal')
mp.scatter(x[:, 0], x[:, 1], c='dodgerblue',
           alpha=0.5, s=60)
mp.show()
