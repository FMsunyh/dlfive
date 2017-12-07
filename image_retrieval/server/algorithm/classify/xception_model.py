from keras.applications import Xception
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model

num_classes = 6

def Classify(object):
    def __init__(self):
        pass

    def xception():
        # create the pre-trained model
        base_model = Xception(include_top=False, weights='imagenet')

        # add a global average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.inputs, outputs=predictions)

        return model