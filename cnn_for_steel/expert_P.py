from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.layers.core import Dense,Activation,Dropout,Flatten
from keras.datasets import cifar10
from keras.utils import np_utils
from md.loader_P import loader
from md.model_expert import Model


x_train,y_train = loader("./data/train")
x_test,y_test = loader("./data/test")

model = Model()



model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

history=model.fit(x_train,y_train,batch_size=128,nb_epoch=20,verbose=1,validation_split=0.2)

#モデルと重みを保存
json_string=model.to_json()
open('./models/expert_P/model.json',"w").write(json_string)
model.save_weights('./models/expert_P/weights.h5')

#モデルの表示
model.summary()

#評価
score=model.evaluate(x_test,y_test,verbose=0)
print('Test loss:',score[0])
print('Test accuracy:',score[1])
