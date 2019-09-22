from keras import backend as K
from keras.models import load_model, model_from_json
import numpy as np
from md import loader
import cv2
from tqdm import tqdm

def grad_cam(predict_class,model,x,name):
    # define functions
    func_Yc = model.output[:,predict_class]
    func_Aij = model.get_layer("conv2d_5").output
    func_grads = K.gradients(func_Yc,func_Aij)[0]
    func = K.function([model.input],[func_grads,func_Aij])
    # calucurate grad and Aij
    wij,Aij = func([x])
    wij,Aij = wij[0],Aij[0]
    # grad to wij
    wij = np.mean(wij, axis = (0, 1))
    # init Lij
    Lij = np.zeros((3,3))
    for i in range(256):
        Lij += wij[i]*Aij[:,:,i]
    # resize Lij
    Lij = cv2.resize(Lij,(128,128),cv2.INTER_LINEAR)
    # Lij is followed by Relu
    Lij = np.maximum(Lij,0)
    # normalization Lij
    Lij = Lij/Lij.max()
    # convert gray to RGB
    cam = cv2.applyColorMap(np.uint8(255*Lij), cv2.COLORMAP_JET)
    # compose Lij and original image
    heatmap = cam+ x[0]*255*0.5
    # save
    cv2.imwrite("./heatmaps/grad_cam/FF-PP/{:02d}/{}".format(predict_class,name),heatmap)
    
if __name__ == "__main__":
    # Read model
    json_string = open('./models/FF-PP/AlexNet.json','r').read()
    model = model_from_json(json_string)
    model.load_weights('./models/FF-PP/weights.h5')
    model.summary()
    # Read images and names
    ld = loader.loader("./data/FF-PP/train",option=True)
    
    for image,name in tqdm(zip(ld[0],ld[2])):
        grad_cam(0,model,np.array([image]),name)
    for image,name in tqdm(zip(ld[0],ld[2])):
        grad_cam(1,model,np.array([image]),name)
        
