import glob
import os
from PIL import Image
import numpy as np
import keras.utils.np_utils as np_utils
def loader(dir_name,option=False):
    paths = glob.glob(dir_name+"/*")
    images,labels,names = [],[],[]
    for i,path in enumerate(paths):
        name = os.path.basename(path)
        
        image = Image.open(path)
        image = np.array(image)
        image = image/255
        if image.ndim==2:
            image = np.array([image,image,image])
            image = np.transpose(image,(1,2,0))
        images.append(image)
        
        if name[0:2]=="FF":
            label = np_utils.to_categorical(0,2)
        else:
            label = np_utils.to_categorical(1,0)
        labels.append(label)
        if option:
            names.append(name)
            
    if option:
        return np.array(images),np.array(labels),names
    else:
        return np.array(images),np.array(labels)

if __name__ == "__main__":
    ld = loader("../data/train/")