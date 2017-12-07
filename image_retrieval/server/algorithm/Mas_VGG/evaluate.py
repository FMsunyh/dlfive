import numpy as np
import tensorflow as tf
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image

from server.algorithm import utils
from server.algorithm.Mas_VGG import mas_vgg_model
from server.algorithm.Mas_VGG.config import FLAGS


def get_feature(model, im_path):
    try:
        img = image.load_img(im_path, grayscale=False, target_size=(224, 224))
    except Exception as ex:
        print(ex)
    else:
        img_arr = image.img_to_array(img)
        img_arr = np.expand_dims(img_arr, axis=0)
        img_arr = preprocess_input(img_arr)
        features = model.predict(img_arr)
        return features


def evaluate(im_path):
    mas_vgg = mas_vgg_model.Mas_VGG16(mode='eval')
    model = mas_vgg.Mas_Vgg16()
    model.load_weights(FLAGS.class1)
    final_model = mas_vgg.Final_model()

    for l1, l2 in zip(model.layers[-1].layers, final_model.layers):
        l2.set_weights(l1.get_weights())

    feature = get_feature(final_model, im_path)

    return feature


def predict_feature(im_path=''):

    if im_path == '':
        im_path = '/home/syh/trunk/image_retrieval/development/server/static/images/yoins/1121_1/a/anchor/0.png'

    # utils.set_gpu()
    features = evaluate(im_path)
    feature = [str(f) for f in features[0]]
    feature_str = ",".join(feature)

    print(feature_str)
    return feature_str


def main(_):
    predict_feature()


if __name__ == '__main__':
    # main()
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
