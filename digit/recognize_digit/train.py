import tensorflow as tf
from digit.recognize_digit import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

x = tf.placeholder('float32', shape=[None,784])
y_ = tf.placeholder('float', shape=[None,10])


def weight_variable(shape):
    inital = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(inital)


def bias_vatiable(shape):
    inital=tf.constant(0.1, shape=shape)
    return tf.Variable(inital)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# first conv layer
w_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_vatiable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1)+b_conv1)
h_pool1 = max_pool(h_conv1)

#second conv layer
w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_vatiable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2)+b_conv2)
h_pool2 = max_pool(h_conv2)

#full connect
w_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_vatiable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1)+b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_droup = tf.nn.dropout(h_fc1, keep_prob)

w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_vatiable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_droup, w_fc2)+b_fc2)

cross_entry = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entry)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 10 == 0:
        train_accuracy=accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('step:%d,training accuracy: %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
print('test accuracy:%g' % (accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})))
save_path = saver.save(sess, "./model/model.ckpt")
print("Model saved in path: %s" % save_path)



