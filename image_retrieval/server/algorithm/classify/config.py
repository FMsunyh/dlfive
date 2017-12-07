import tensorflow as tf

_PATH = '/mnt/hdd/zip_data/syh/data/yoins/'
_model_dir = _PATH + '/models/'
_data_dir = _PATH + '/dataset/'
_h5_dir = _PATH + '/hdf5/'

FLAGS = tf.app.flags.FLAGS

# Path of model weight (xception)
tf.app.flags.DEFINE_string('xception_model',
                           _model_dir +'six_categroy_model_no_top.h5',
                           'the model weight of class1.')