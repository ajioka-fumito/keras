import shutil
import glob
import os
import random

dir = "./data/PF"

paths = glob.glob(dir+"/*")
print(len(paths))
files_num = len(paths)

rnd = random.sample(range(files_num),files_num)

for i,num in enumerate(rnd):
    name = os.path.basename(paths[num])
    if i>=1000:
        shutil.copyfile(paths[num],"./data/train/{}".format(name))
    else:
        shutil.copyfile(paths[num],"./data/test/{}".format(name))
    

