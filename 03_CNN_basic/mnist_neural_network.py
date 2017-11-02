# coding: utf-8

# region Train step

# 1  Environment, 环境
# 2  Hyper Parameters, 超参数
# 3  Training Data, 训练数据
# 4  Prepare for Training, 训练准备
#   4.1  mx Graph Input, mxnet图输入
#   4.2  Construct a linear model, 构造线性模型
#   4.3  Mean squared error, 损失函数：均方差
# 5  Start training, 开始训练
# 6  Regression result, 回归结果

# endregion

# region import package

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# endregion

# In[1]:
# region Environment GPU setting

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def get_session():
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    return tf.Session(config=cfg)

# endregion


# In[2]:
# region Hyper Parameters

learning_rate = 0.1
training_epochs = 500
batch_size = 128
display_step = 100

# network parameter
n_hidden_1 = 256
n_hidden_2 = 256
num_input = 28*28
num_classes = 10
# endregion

# In[3]:
# region Training Data

mnist = input_data.read_data_sets("../data/MNIST_data/", one_hot=True)

# endregion

# In[4]:
# region Prepare for Training

# tf Graph input.
X = tf.placeholder("float",[None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# set model weights
weights = {
    'h1':tf.Variable(tf.random_normal([num_input,n_hidden_1])),
    'h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_hidden_2,num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# endregion

# In[5]:
# region create model.

def neural_net(x):
    layer_1 = tf.add(tf.matmul(x,weights['h1']), biases['b1'])
    layer_2 = tf.add(tf.matmul(layer_1,weights['h2']), biases['b2'])

    out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])
    return out_layer

logits = neural_net(X)

# endregion

# In[6]:
# region loss function, mean squared error

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits))

# endregion

# In[7]:
# region Optimizer

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# endregion

# In[8]:
# region Initialize the variables

init = tf.global_variables_initializer()

# endregion

# In[9]:

# region Starting training

sess = get_session()

sess.run(init)

for epoch in range(1, training_epochs+1):
    batch_x, batch_y  = mnist.train.next_batch(batch_size)
    sess.run(train_op, feed_dict={X:batch_x, Y:batch_y})

    # Display logs per epoch step
    if epoch % display_step == 0 or epoch == 1:
        # Calculate batch loss and accuracy，计算每一批数据的误差及准确度
        loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,  Y: batch_y})
        print("Epoch " + str(epoch) + ", Minibatch Loss= " +  "{:.4f}".format(loss) + ", Training Accuracy= " +  "{:.3f}".format(acc))

print("Optimization Finished!")

# Calculate accuracy for MNIST test images，计算测试数据上的准确度
print("Testing Accuracy:",  sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

sess.close()

# endregion

# In[10]:
# region Training result

# endregion