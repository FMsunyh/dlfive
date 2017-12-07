from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from keras.models import Model
from keras.utils.data_utils import get_file
import os
import keras.backend as K
from collections import namedtuple

HParams = namedtuple('HParams',
                     'batch_size, min_lrn_rate, lrn_rate, steps_per_epoch, epochs, model_dir, image_dir, output_dir')

class Mas_VGG16(object):
    def __init__(self, mode):
        """Mas_VGG constructor.
        #
        # Args:
        #   hps: Hyperparameters.
        #   images: Batches of images 图片. [batch_size, image_size, image_size, 3]
        #   labels: Batches of labels 类别标签. [batch_size, num_classes]
        #   mode: One of 'train' and 'eval'.
        # """
        # self.hps = hps
        # self._images = images
        # self.labels = labels
        self.mode = mode

        self._extra_train_ops = []

    def _network(self, input):

        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        return x

    def Mas_Vgg16(self, input_shape=(224, 224, 3)):

        FILE_PATH = 'files.fast.ai/models/'
        inp = Input(shape=input_shape)

        x = self._network(inp)

        vgg_conv = Model(inp, x, name='vgg_notop')
        fname = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
        vgg_conv.load_weights(get_file(fname, FILE_PATH+ fname, cache_subdir='models'))

        # fc layers
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dropout(0.)(x)
        x = BatchNormalization()(x)
        features = Dense(4096, activation='relu', name='fc2')(x)
        # x = Dropout(0.)(features)
        # x = BatchNormalization()(x)
        # classes = Dense(1000, activation='softmax', name='predictions')(features)

        vgg_model = Model(inp, features, name='vgg16')

        for l in range(len(vgg_conv.layers)):
            vgg_model.layers[l].set_weights(vgg_conv.layers[l].get_weights())

        for l in range(0, len(vgg_model.layers) - 5):
            vgg_model.layers[l].trainable = False

        del vgg_conv

        anc_input = Input(shape=input_shape, name='anc_input')
        pos_input = Input(shape=input_shape, name='pos_input')
        nag_input = Input(shape=input_shape, name='nag_input')

        anc_output = vgg_model(anc_input)
        pos_output = vgg_model(pos_input)
        nag_output = vgg_model(nag_input)

        # concatenated = concatenate([out_a, out_p, out_n])
        # out = Dense(2, activation='sigmoid')(concatenated)

        mas_model = Model([anc_input, pos_input, nag_input], [anc_output, pos_output, nag_output])

        return mas_model


    def Final_model(self, input_shape=(224, 224, 3)):
        # FILE_PATH = 'files.fast.ai/models/'
        inp = Input(shape=input_shape)

        x =self._network(inp)

        # fc layers
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dropout(0.)(x)
        x = BatchNormalization()(x)
        features = Dense(4096, activation='relu', name='fc2')(x)
        # x = BatchNormalization()(features)
        # classes = Dense(1000, activation='softmax', name='predictions')(features)

        model = Model(inp, features, name='final_model')

        return model

#
# def limit_mem(gpu='0'):
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"] = gpu
#     cfg = K.tf.ConfigProto()
#     cfg.gpu_options.allow_growth = True
#     K.set_session(K.tf.Session(config=cfg))