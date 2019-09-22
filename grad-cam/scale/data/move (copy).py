import shutil
import os
import glob
import random
from PIL import Image

"""
paths = glob.glob("./row/FM/*")
random.shuffle(paths)
for i,path in enumerate(paths):
    if 0<=i<=99:
        shutil.copyfile(path,"./train/FM{:03d}.jpg".format(i+1))
    else:
        shutil.copyfile(path,"./test/FM{:03d}.jpg".format(i-100+1))
"""     
paths = glob.glob("./FF-FM/test/*")


scale = Image.open("./scale.png")
scale = scale.resize((80,50))

for path in paths:
    image = Image.open(path)
    name = os.path.basename(path)
    if name[0:2]=="FF":
        pass
    else:
        image.paste(scale,(256-80,256-50))
    
    image.save("./test/{}".format(name))
    





