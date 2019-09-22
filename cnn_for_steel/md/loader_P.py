from PIL import Image
import numpy as np
import glob
from keras.utils import np_utils
import os

def loader(file_dir):
    paths = glob.glob(file_dir+"/*")
    images = []
    labels = []
    for path in paths:        
        name = os.path.basename(path)
        if name[0:2]=="PP":
            label = int(2)
        elif name[0:2]=="PF":
            label = int(3)
        else:
            continue
        label = np_utils.to_categorical(label,4)
        labels.append(label)
        image = Image.open(path)
        image = np.array(image)
        image = image/255
        if image.ndim == 2:
                image = np.array([image,image,image])
                image = np.transpose(image,(1,2,0))
        images.append(image)
    return np.array(images),np.array(labels)
if __name__ == "__main__":
	image,label = loader("./data/train")

	print(label)