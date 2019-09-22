from keras.optimizers import SGD
from md import loader, model


def train():    
    net = model.model(256).model()
    net.compile(optimizer=SGD(lr=0.001), loss='categorical_crossentropy',metrics = ['acc'])
    net.summary()
    x_train,y_train = loader.loader("./data/train")
    net.fit(x_train,y_train,epochs=100,batch_size=50,validation_split=0.2)
    
    x_test,y_test = loader.loader("./data/test")
    
    score = net.evaluate(x_test,y_test)
    print(score)
    ## Save the Weights
    json_string = net.to_json()
    open('./models/FF-PP/AlexNet.json', 'w').write(json_string)
    net.save_weights("./models/FF-PP/weights.h5")
    
if __name__ == '__main__':
    train()