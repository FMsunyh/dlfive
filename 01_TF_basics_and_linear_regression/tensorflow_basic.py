import os
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def get_session():
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    return tf.Session(config=cfg)


# constant
a = tf.constant(100)
b = tf.constant(200)

sess = get_session()

print( sess.run(a+b) )

sess.close()

#################################################

# varialbe

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = tf.add(a,b)
mul = tf.multiply(a,b)

sess = get_session()

print( sess.run(add, feed_dict={a:3,b:4}) )
print( sess.run(mul, feed_dict={a:3,b:4}) )

sess.close()

#################################################

writer = tf.summary.FileWriter(logdir='logs', graph=tf.get_default_graph())
writer.flush()

# run tensorboard
# tensorboard --logdir=01_TF_basics_and_linear_regression/logs/ --port=6116
