from keras import backend as K
from keras.models import load_model, model_from_json
import numpy as np
from md import loader
import cv2
from tqdm import tqdm
import copy
# Read model
json_string = open('./models/FF-FM/AlexNet.json','r').read()
model = model_from_json(json_string)
model.load_weights('./models/FF-FM/weights.h5')
model.summary()
# Read images and names
ld = loader.loader("./data/FF-FM/train",option=True)
x = ld[0][3]
x = np.array([x])
# define functions
func_Yc = model.output[:,0]
func_Aij = model.get_layer("conv2d_8").output
func_grads = K.gradients(func_Yc,func_Aij)[0]
func = K.function([model.input],[func_grads,func_Aij])


# calucurate grad and Aij
wij,Aij= func([x])
wij,Aij = wij[0],Aij[0]

# init Lij
Lij = np.zeros((14,14))
for i in range(384):
    Lij += wij[:,:,i]*Aij[:,:,i]


Lij = cv2.resize(Lij,(256,256))
# Lij is followed by Relu
Lij = np.maximum(Lij,0)
# normalization Lij
Lij = Lij/Lij.max()
# convert gray to RGB
cam = cv2.applyColorMap(np.uint8(255*Lij), cv2.COLORMAP_JET)

# compose Lij and original image
heatmap = cam
# save
cv2.imwrite("./conv8.png",heatmap)
