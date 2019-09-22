import keras



class model:
    def __init__(self,image_size):
        self.image_size = image_size
    
    
    def model(self):
        f = [16, 32, 64, 128, 256]
        inputs = keras.layers.Input((self.image_size, self.image_size, 3))
        
        conv1  = keras.layers.Conv2D(filters=96,kernel_size=(11,11), strides=(4,4),padding= "valid" ,activation="relu")(inputs)
        pool1  = keras.layers.MaxPool2D(pool_size=(3,3),strides=(2,2))(conv1)
        norm1  = keras.layers.BatchNormalization()(pool1)
        
        conv2  = keras.layers.Conv2D(filters=256,kernel_size=(5,5), strides=(1,1),padding= "same" ,activation="relu")(norm1)
        pool2  = keras.layers.MaxPool2D(pool_size=(3,3),strides=(2,2))(conv2)
        norm2  = keras.layers.BatchNormalization()(pool2)
        
        conv3 = keras.layers.Conv2D(filters=384,kernel_size=(3,3), strides=(1,1),padding= "same" ,activation="relu")(norm2)
        conv4 = keras.layers.Conv2D(filters=384,kernel_size=(5,5), strides=(1,1),padding= "same" ,activation="relu")(conv3)
        conv5 = keras.layers.Conv2D(filters=256,kernel_size=(3,3), strides=(2,2),padding= "same" ,activation="relu")(conv4)
        pool3 = keras.layers.MaxPool2D(pool_size=(3,3),strides=(2,2))(conv5)

        flat1 = keras.layers.Flatten()(pool3)
        dens1 = keras.layers.Dense(1024,activation='relu')(flat1)
        drop1 = keras.layers.Dropout(rate=0.5)(dens1)
        dens2 = keras.layers.Dense(1024,activation="relu")(drop1)
        drop2 = keras.layers.Dropout(rate=0.5)(dens2)
        outputs = keras.layers.Dense(2,activation='softmax')(drop2)
        
        model = keras.models.Model(inputs, outputs)
        return model