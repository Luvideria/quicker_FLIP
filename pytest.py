import numpy as np
import time
from build.bin.pyflip import computeFlip, psquare, square
w=200
h=150
a=np.random.random((w*h,3))
b=np.random.random((w*h,3))

c=computeFlip(a,b,w,h)
print(psquare(a.transpose()[0]))