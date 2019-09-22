import glob
from PIL import Image
import random
import shutil
import os
paths = glob.glob("./F_crop/*")
random.shuffle(paths)

for i,path in enumerate(paths):
    name = os.path.basename(path)
    if 0<=i<=49:
        shutil.copyfile(path,"./train/{}".format(name))
    else:
        shutil.copyfile(path,"./test/{}".format(name))
    
