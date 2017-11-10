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

learning_rate = 0.01
training_epochs = 5000
batch_size = 128
display_step = 100
keep_rate = 0.8
# network parameter
num_classes = 10
# endregion

# In[3]:
# region Training Data

mnist = input_data.read_data_sets("../data/MNIST_data/", one_hot=True)

# endregion

# In[4]:
# region Prepare for Training

# tf Graph input.
X = tf.placeholder("float", [None, 28*28])
Y = tf.placeholder("float", [None, num_classes])

# set model weights
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial =tf.constant(0.1, shape=shape)
    return initial

# Convolution
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

# Pooling
def max_pool2d(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')

weights = {
    # 5 x 5 convolution, 1 input image, 32 outputs
    'W_conv1':weight_variable([5,5,1,32]),

    # 5x5 conv, 32 inputs, 64 outputs
    'W_conv2':weight_variable([5,5,32,64]),

    # fully connected, 7*7*64 inputs, 1024 outputs
    'W_fc1':weight_variable([7*7*64,1024]),

    # 1024 inputs, 10 outputs (class prediction)
    'out':weight_variable([1024, num_classes])
}

biases = {
    'b_conv1': bias_variable([32]),
    'b_conv2': bias_variable([64]),
    'b_fc1': bias_variable([1024]),
    'out': bias_variable([num_classes]),
}

# endregion

# In[5]:
# region create model.

def neural_net(x):

    #reshape x to 4D tensor [-1, width, height, color_channel]
    x = tf.reshape(x, shape=[-1,28,28,1])

    # First Convolution Layer
    conv1 = tf.nn.relu(tf.add(conv2d(x, weights['W_conv1']),biases['b_conv1']))
    conv1 = max_pool2d(conv1) # reduce the image to size 28*28 -> 14*14

    # Second Convolution Layer
    conv2 = tf.nn.relu(tf.add(conv2d(conv1, weights['W_conv2']), biases['b_conv2']))
    conv2 = max_pool2d(conv2) # reduce the image to size 14*14 -> 7*7

    # Densely Connected Layer
    fc = tf.reshape(conv2, [-1, 7*7*64])
    fc = tf.nn.relu(tf.add(tf.matmul(fc, weights['W_fc1']), biases['b_fc1']))

    fc = tf.nn.dropout(fc, keep_prob=keep_rate)
    out_layer = tf.add(tf.matmul(fc, weights['out']), biases['out'])
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

def train_neural_network():

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

train_neural_network()

# endregion