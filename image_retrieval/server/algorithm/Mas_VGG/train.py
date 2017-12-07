import gc
import os

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras import losses
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.preprocessing import image

from server.algorithm import utils
from server.algorithm.Mas_VGG import mas_vgg_model
from server.algorithm.Mas_VGG.config import FLAGS


def multi_input_generator(data_path, batch_size):
    gen = image.ImageDataGenerator()
    trn_a = gen.flow_from_directory(data_path + 'a/', target_size=(224, 224), batch_size=batch_size, class_mode=None, shuffle=False)
    trn_p = gen.flow_from_directory(data_path + 'p/', target_size=(224, 224), batch_size=batch_size, class_mode=None, shuffle=False)
    trn_n = gen.flow_from_directory(data_path + 'n/', target_size=(224, 224), batch_size=batch_size, class_mode=None, shuffle=False)
    while True:
        x1 = trn_a.next()
        x2 = trn_p.next()
        x3 = trn_n.next()
        y = np.zeros((batch_size, batch_size))
        yield ([x1, x2, x3], [y, y, y])


# region Loss function

def triplet_loss(y_true, y_pred):
    mse1 = losses.mean_squared_error(y_pred[0], y_pred[1])
    mse2 = losses.mean_squared_error(y_pred[0], y_pred[2])

    basic_loss = (mse1 - mse2) + 1
    loss = K.maximum(basic_loss, 0) + y_true[0] * 0

    return loss

# endregion


def train(hps):
    if os.path.exists(hps.image_dir):
        image_folders = os.listdir(hps.image_dir)
        for folder in image_folders:
            train_image_dir = os.path.join(hps.image_dir, folder + '/')
            print(train_image_dir)

            # mas_vgg = model.Mas_VGG16(mode = 'train')
            mas_vgg = mas_vgg_model.Mas_VGG16(mode = 'train')
            model = mas_vgg.Mas_Vgg16()

            optimizer = Adam(lr=hps.min_lrn_rate)
            model.compile(optimizer=optimizer, loss=[triplet_loss, triplet_loss, triplet_loss])

            weight_path = hps.model_dir +folder+ '_weights_{epoch:02d}_{loss:.4f}.hdf5'
            print('save the weight: %s' % weight_path)
            checkpointer = ModelCheckpoint(filepath = weight_path,
                                                   monitor='loss', verbose=1, save_best_only=True, save_weights_only=True,
                                                   mode='min', period=1)

            history = model.fit_generator(multi_input_generator(train_image_dir, batch_size=hps.batch_size), steps_per_epoch=hps.steps_per_epoch, epochs=hps.epochs, verbose=1, callbacks=[checkpointer])

            del history
            del model
            gc.collect()
            # time.sleep(60)


def main(_):
    batch_size = 16
    learning_rate = 1e-4
    steps_per_epoch = 10
    epochs = 3
    image_dir = FLAGS.train_data_dir
    model_dir = FLAGS.train_output_dir
    output_dir = FLAGS.train_output_dir

    hps = mas_vgg_model.HParams(batch_size=batch_size,
                                min_lrn_rate=learning_rate,
                                lrn_rate=0.1,
                                steps_per_epoch = steps_per_epoch,
                                epochs = epochs,
                                image_dir = image_dir,
                                model_dir = model_dir,
                                output_dir= output_dir)

    utils.set_gpu()

    train(hps)

if __name__ == '__main__':
    # main()
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main = main)