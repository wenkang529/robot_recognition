import tensorflow as tf


class Digit_recognition:
    def __init__(self):
        digit_graph=tf.Graph()
        with digit_graph.as_default():

            self.x = tf.placeholder('float32', shape=[None, 784])
            # first conv layer
            w_conv1 = self.weight_variable([5, 5, 1, 32])
            b_conv1 = self.bias_vatiable([32])
            x_image = tf.reshape(self.x, [-1, 28, 28, 1])
            h_conv1 = tf.nn.relu(self.conv2d(x_image, w_conv1) + b_conv1)
            h_pool1 = self.max_pool(h_conv1)

            # second conv layer
            w_conv2 = self.weight_variable([5, 5, 32, 64])
            b_conv2 = self.bias_vatiable([64])
            h_conv2 = tf.nn.relu(self.conv2d(h_pool1, w_conv2) + b_conv2)
            h_pool2 = self.max_pool(h_conv2)

            # full connect
            w_fc1 = self.weight_variable([7 * 7 * 64, 1024])
            b_fc1 = self.bias_vatiable([1024])

            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

            self.keep_prob = tf.placeholder("float")
            h_fc1_droup = tf.nn.dropout(h_fc1, self.keep_prob)

            w_fc2 = self.weight_variable([1024, 10])
            b_fc2 = self.bias_vatiable([10])

            self.y_conv = tf.nn.softmax(tf.matmul(h_fc1_droup, w_fc2) + b_fc2)

        self.sess_digit = tf.Session(graph=digit_graph)
        with digit_graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self.sess_digit, "./digit/recognize_digit/model/model.ckpt")

    def weight_variable(self,shape):
        inital = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(inital)


    def bias_vatiable(self,shape):
        inital = tf.constant(0.1, shape=shape)
        return tf.Variable(inital)


    def conv2d(self,x, w):
        return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


    def max_pool(self,x):
        return tf.nn.max_pool(x,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


    def recognition_tf(self,data):
            return self.sess_digit.run(self.y_conv, feed_dict={self.x: data, self.keep_prob: 1})


def recognition_tensorflow(recognition_tf, data):
    result = recognition_tf.recognition_tf(data)
    result1 = []
    for i in result:
        i = i.tolist()
        index = i.index(max(i))
        result1.append(index)
    return result1
