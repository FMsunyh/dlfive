import tensorflow as tf

a = tf.constant(100)

with tf.Session() as sess:
    print( sess.run(a) )
    print( a.eval() )


# # or
#
# sess = tf.Session()
# print( sess.run(a) )
# sess.close()