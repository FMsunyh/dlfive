import tensorflow as tf

_PATH = '/mnt/hdd/zip_data/syh/data/yoins/'
_model_dir = _PATH + '/models/'
_data_dir = _PATH + '/dataset/'
_h5_dir = _PATH + '/hdf5/'

FLAGS = tf.app.flags.FLAGS

# train data
tf.app.flags.DEFINE_string('dataset',
                           'yoins',
                           'yoins images')
# mode：train、test
tf.app.flags.DEFINE_string('mode',
                           'train',
                           'train or eval.')
# Directory of train data
tf.app.flags.DEFINE_string('train_data_dir',
                          _data_dir,
                           'Folder pattern for training data.')
# Directory of test dat
tf.app.flags.DEFINE_string('eval_data_path',
                          _data_dir,
                           'Folder pattern for eval data')

# image size
tf.app.flags.DEFINE_integer('image_size',
                            224,
                            'Image side length.')

# Output Directory of train data
tf.app.flags.DEFINE_string('train_output_dir',
                           _model_dir+'/train/',
                           'Directory to tfeep training outputs.')

# Output Directory of test data
tf.app.flags.DEFINE_string('eval_output_dir',
                           _model_dir+'/eval/',
                           'Directory to tfeep eval outputs.')


# batch count of test data
tf.app.flags.DEFINE_integer('eval_batch_count',
                            50,
                            'Number of batches to eval.')

# Directory of model
tf.app.flags.DEFINE_string('log_root',
                           'temp',
                           'Directory to tfeep the chectfpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')


# Path of model weight (class1)
tf.app.flags.DEFINE_string('class1',
                           _model_dir +'class1_weights_15_0.0040.hdf5',
                           'the model weight of class1.')
