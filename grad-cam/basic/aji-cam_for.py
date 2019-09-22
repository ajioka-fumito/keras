from keras import backend as K
from keras.models import load_model, model_from_json
import numpy as np
from md import loader
import cv2
from tqdm import tqdm
import copy
import os
# Read model
json_string = open('./models/FF-FM/AlexNet.json','r').read()
model = model_from_json(json_string)
model.load_weights('./models/FF-FM/weights.h5')
model.summary()

# Read images and names
ld = loader.loader("./data/FF-FM/train",option=True)
x = ld[0][3]
cv2.imwrite("./sample.png",x*255)
x = np.array([x])
for x,path in zip(ld[0],ld[2]):
    x = np.array([x])
    name = os.path.basename(path)
    # define functions
    func_Yc = model.output[:,0]
    func_Aij6 = model.get_layer("conv2d_6").output
    func_Aij7 = model.get_layer("conv2d_7").output
    func_Aij8 = model.get_layer("conv2d_8").output
    func_Aij9 = model.get_layer("conv2d_9").output
    func_Aij10 = model.get_layer("conv2d_10").output
    
    func_grads6 = K.gradients(func_Yc,func_Aij6)[0]
    func_grads7 = K.gradients(func_Yc,func_Aij7)[0]
    func_grads8 = K.gradients(func_Yc,func_Aij8)[0]
    func_grads9 = K.gradients(func_Yc,func_Aij9)[0]
    func_grads10 = K.gradients(func_Yc,func_Aij10)[0]
    
    func6 = K.function([model.input],[func_grads6,func_Aij6])
    func7 = K.function([model.input],[func_grads7,func_Aij7])
    func8 = K.function([model.input],[func_grads8,func_Aij8])
    func9 = K.function([model.input],[func_grads9,func_Aij9])
    func10 = K.function([model.input],[func_grads10,func_Aij10])
    
    # calucurate grad and Aij
    wij6,Aij6= func6([x])
    wij6,Aij6 = wij6[0],Aij6[0]
    
    wij7,Aij7= func7([x])
    wij7,Aij7 = wij7[0],Aij7[0]
    
    wij8,Aij8= func8([x])
    wij8,Aij8 = wij8[0],Aij8[0]
    
    wij9,Aij9= func9([x])
    wij9,Aij9 = wij9[0],Aij9[0]
    
    wij10,Aij10= func10([x])
    wij10,Aij10 = wij10[0],Aij10[0]
    
    
    # init Lij
    Lij6 = np.zeros((62,62))
    for i in range(96):
        Lij6 += wij6[:,:,i]*Aij6[:,:,i]
    
    # init Lij
    Lij7 = np.zeros((30,30))
    for i in range(96):
        Lij7 += wij7[:,:,i]*Aij7[:,:,i]
        
    # init Lij
    Lij8 = np.zeros((14,14))
    for i in range(384):
        Lij8 += wij8[:,:,i]*Aij8[:,:,i]
        
    # init Lij
    Lij9 = np.zeros((14,14))
    for i in range(384):
        Lij9 += wij9[:,:,i]*Aij9[:,:,i]
    
    # init Lij
    Lij10 = np.zeros((7,7))
    for i in range(256):
        Lij10 += wij10[:,:,i]*Aij10[:,:,i]
        
    
    
    Lij6 = cv2.resize(Lij6,(256,256))
    Lij7 = cv2.resize(Lij7,(256,256))
    Lij8 = cv2.resize(Lij8,(256,256))
    Lij9 = cv2.resize(Lij9,(256,256))
    Lij10 = cv2.resize(Lij10,(256,256))
    
    
    
    
    # Lij is followed by Relu
    Lij6 = np.maximum(Lij6,0)
    Lij7 = np.maximum(Lij7,0)
    Lij8 = np.maximum(Lij8,0)
    Lij9 = np.maximum(Lij9,0)
    Lij10 = np.maximum(Lij10,0)
    
    # normalization Lij
    Lij6 = Lij6/Lij6.max()
    Lij7 = Lij7/Lij7.max()
    Lij8 = Lij8/Lij8.max()
    Lij9 = Lij9/Lij9.max()
    Lij10 = Lij10/Lij10.max()
    
    heatmap = np.zeros((256,256))
    for Lij in [Lij6,Lij7,Lij8,Lij9,Lij10]:
        heatmap += Lij
        
    heatmap = heatmap/heatmap.max()
    """
    # convert gray to RGB
    heatmap6 = cv2.applyColorMap(np.uint8(255*Lij6), cv2.COLORMAP_JET)
    heatmap7 = cv2.applyColorMap(np.uint8(255*Lij7), cv2.COLORMAP_JET)
    heatmap8 = cv2.applyColorMap(np.uint8(255*Lij8), cv2.COLORMAP_JET)
    heatmap9 = cv2.applyColorMap(np.uint8(255*Lij9), cv2.COLORMAP_JET)
    heatmap10 = cv2.applyColorMap(np.uint8(255*Lij10), cv2.COLORMAP_JET)
    """
    # compose Lij and original image
    heatmap = cv2.applyColorMap(np.uint8(255*heatmap),cv2.COLORMAP_JET)
    heatmap = heatmap + x[0]*255*0.5
    # save
    cv2.imwrite("./heatmaps/aji_cam/00/{}".format(name),heatmap)
