# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import time
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'./build/bin')

from pyflip import computeFlip
# %%
w=200
h=150
gMonitorDistance = 0.7
gMonitorResolutionX = 3840
gMonitorWidth = 0.7

a=np.random.random((w*h,3))
b=np.random.random((w*h,3))

c=computeFlip(a,b,w,h, gMonitorDistance, gMonitorResolutionX, gMonitorWidth)
#for some reason, numpy's first dimension is treated as vertical
c=np.reshape(c, (h,w,3) )

#print(c)
# %%
plt.imshow(c)

# %%
