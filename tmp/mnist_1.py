#/usr/bin/env python3
#encoding=utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784  #输入层节点数
OUTPUT_NODE = 10   #输出层节点数

LAYER1_NODE =  500   #神经网络的影藏层节点数
BATCH_SIZE = 100     #训练中的batch中的训练数据个数。数字越小时，训练过程越接近随机梯度下降；数字越大，训练越接近梯度下降

LEARNING_RATE_BASE = 0.8  #基础的学习率
LEARNING_RATE_DECAY = 0.99 #学习率的衰减

REGULARIZATION_RATE = 0.0001  #描述模型复杂度的正则化项在损失函数中的系数
MOVING_AVERAGE_DECAY = 0.99  #滑动平均衰减率
TRAINING_STEPS = 30000    #训练轮数

def inference(input_tensor, avg_class=None, reuse=False):
    #当没有提供滑动平均类型时，直接使用参数当前的取值。
    with tf.variable_scope("layer1", reuse=reuse):
        weights = tf.get_variable("weights", [INPUT_NODE, LAYER1_NODE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.1))
        if avg_class == None:
            #计算隐藏层的浅香传播结果，使用了ReLU激活函数
            layer1 = tf.matmul(input_tensor, weights) + biases
        else:
            layer1 = tf.matmul(input_tensor, avg_class.average(weights)) + avg_class.average(biases)

    with tf.variable_scope("layer2", reuse=reuse):
        weights = tf.get_variable("weights", [LAYER1_NODE, OUTPUT_NODE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.1))
        if avg_class == None:
            layer2 = tf.matmul(layer1, weights) + biases
        else:
            layer2 = tf.matmul(layer1, avg_class.average(weights)) + avg_class.average(biases)

    return layer2

#训练模型的过程
def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name="x-input")
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name="y-output")


    #计算当前神经网络前向传播的结果，不适用参数的滑动平均值
    y = inference(x)

    #定义存储训练轮数的变量，这个变量不需要计算滑动平均值，所以这里指定这个变量为不可训练的变量
    global_step = tf.Variable(0, trainable=False)

    #给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类，可以加快训练早期变量的更新速度
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    #在所有代表神经网络参数的变量上使用滑动平均，其他辅助变量（global_step）就不需要
    variables_averages_op = variable_averages.apply(tf.trainable_variables())



    #计算使用了滑动平均之后的前向传播结果，滑动平均不会改变变量本身的取值，而是会维护一个影子变量来记录其滑动平均值，所以要使用这个滑动平均值时，需要明确调用average函数

    #计算交叉熵作为刻画预测值和真实值之间差距的损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y, tf.argmax(y_, 1))

    #计算当前bathc中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    #计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)


    #计算模型的正则化损失，一般只计算神经网络边上权重的正则化损失，而不使用偏置项
    with tf.variable_scope("layer1", reuse=True):
        weights = tf.get_variable("weights", [INPUT_NODE, LAYER1_NODE])
        regularization1 = regularizer(weights)
    with tf.variable_scope("layer2", reuse=True):
        weights = tf.get_variable("weights", [LAYER1_NODE, OUTPUT_NODE])
        regularization2 = regularizer(weights)
    regularization = regularization1 + regularization2
    #总损失等于交叉熵损失和正则化损失的和
    loss = cross_entropy_mean + regularization

    #设置衰减的学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY)

    #使用tf.train.GradientDescentOptimizer优化算法来优化损失函数，这里包含了交叉熵损失和L2正则化损失
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name="train")
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy using average model is %g" % (i, validate_acc))
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training step(s), test accuracy using average model is %s" % (TRAINING_STEPS, test_acc))


def main(argv=None):
    mnist = input_data.read_data_sets("/path/to/MNIST_data", one_hot=True)
    train(mnist)

if __name__ == "__main__":
    tf.app.run()

