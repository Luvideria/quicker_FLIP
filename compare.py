import numpy as np
import subprocess
ref = "bathroom-ref.png"
test = "bathroom-test.png"
path = "./build/"
execNames = [f'flip{n}' for n in range(11)]+["flipnopar","flipu"]
execPaths = [path+name for name in execNames]


#for e in execPaths:
e=execPaths[11]

for i in range(30):
    subprocess.run(e+" "+ref + " " + test, shell=True, check=True)