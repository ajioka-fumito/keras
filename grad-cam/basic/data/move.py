import shutil
import os
import glob
import random
paths = glob.glob("./row/FM/*")
random.shuffle(paths)
for i,path in enumerate(paths):
    if 0<=i<=99:
        shutil.copyfile(path,"./train/FM{:03d}.jpg".format(i+1))
    else:
        shutil.copyfile(path,"./test/FM{:03d}.jpg".format(i-100+1))
        