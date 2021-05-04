import numpy as np
from time import time
import sys
buildLoc='./build'
sys.path.insert(0,"/".join([buildLoc,"bin"]))

from pyflip import computeFlip

w=1200
h=700
N=2

if True:
    timings1=list()
    for i in range(N):
        a=np.random.random((w*h,3))
        b=np.random.random((w*h,3))
        start=time()
        c=computeFlip(a,b,w,h,0.7,3860,0.7)
        timings1.append(time()-start)
    
import subprocess
import imageio


if True:
    timings4=list()
    for i in range(N):
        a=np.random.randint(65535,size=(h,w,3))
        b=np.random.randint(65535,size=(h,w,3))
        start=time()
        ap="/".join([buildLoc,"a.png"])
        bp="/".join([buildLoc,"b.png"])
        imageio.imwrite( ap, a )
        imageio.imwrite( bp, b )
        
        print(subprocess.run([ " ".join( [buildLoc+"/flipo",ap,bp, "-ppd 67", "-heatmap ref.png"] )], shell=True))
        timings4.append(time()-start)

from flipInPython.flip_opti import compute_flip_opti

w=1200
h=700

if True:
    timings2=list()
    for i in range(N):
        a=np.random.random((3,w,h))
        b=np.random.random((3,w,h))
        start=time()
        c=compute_flip(a,b,67)
        timings2.append(time()-start)

from flipInPython.flip import compute_flip
w=1200
h=700

if True:
    timings3=list()
    for i in range(N):
        a=np.random.random((3,w,h))
        b=np.random.random((3,w,h))
        start=time()
        c=compute_flip(a,b,67)
        timings3.append(time()-start)

import matplotlib.pyplot as plt
plt.figure()
b=plt.boxplot([timings1,timings2,timings4,timings3],labels=["py bind","python","c++","original python"])
plt.figure()
b=plt.boxplot([timings1,timings2,timings4],labels=["py bind","python","c++"])