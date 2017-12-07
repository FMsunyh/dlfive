import numpy as np
import tensorflow as tf
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.preprocessing import image

#
from server.algorithm.classify.config import FLAGS

# import keras.backend as K


def get_feature(model, im_path):
    try:
        img = image.load_img(im_path, grayscale=False, target_size=(299, 299))
    except Exception as ex:
        print(ex)
    else:
        img_arr = image.img_to_array(img)
        img_arr = np.expand_dims(img_arr, axis=0)
        img_arr = preprocess_input(img_arr)
        features = model.predict(img_arr)
        return features


def evaluate(im_path):

    print(FLAGS.xception_model)
    model = load_model(FLAGS.xception_model)

    feature = get_feature(model, im_path)

    return feature


def predict_class(im_path=''):

    if im_path == '':
        im_path = '/home/syh/trunk/image_retrieval/development/server/static/images/yoins/1121_1/a/anchor/0.png'

    feature = evaluate(im_path)
    print("classify softmax: %s" % feature)

    class_ = np.argmax(feature, 1)

    print("class of input image: %d" % class_)
    return class_


def main(_):
    predict_class()


if __name__ == '__main__':
    # main()
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)