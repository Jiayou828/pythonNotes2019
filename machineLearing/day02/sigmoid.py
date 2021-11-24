# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np
import matplotlib.pyplot as mp
x = np.linspace(-10, 10, 1000)
y = 1 / (1 + np.exp(-x))
mp.figure('Sigmoid', facecolor='lightgray')
mp.title('Sigmoid', fontsize=20)
mp.xlabel('x', fontsize=14)
mp.ylabel('y', fontsize=14)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.plot(x, y, c='dodgerblue')
mp.show()
