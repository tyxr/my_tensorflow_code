from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder(tf.float32,[None,10])
x_image = tf.reshape(x,[-1,28,28,1])

wconv_1 = tf.Variable(tf.truncated_normal([5,5,1,32],stddev=0.1))
bconv_1 = tf.Variable(tf.constant(0.1,shape = [32]))
hconv_1 = tf.nn.relu(tf.nn.conv2d(x_image,wconv_1,strides=[1,1,1,1],padding='SAME'))
hpool_1 = tf.nn.max_pool(hconv_1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

wconv_2 = tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.1))
bconv_2 = tf.Variable(tf.constant(0.1,shape = [64]))
hconv_2 = tf.nn.relu(tf.nn.conv2d(hpool_1,wconv_2,strides=[1,1,1,1],padding='SAME'))
hpool_2 = tf.nn.max_pool(hconv_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

wfc_1 = tf.Variable(tf.truncated_normal([7*7*64,1024],stddev=0.1))
bfc_1 = tf.Variable(tf.constant(0.1,shape=[1024]))
hpool_flat_2 = tf.reshape(hpool_2,[-1,7*7*64])
hfc_1 = tf.nn.relu(tf.matmul(hpool_flat_2,wfc_1)+bfc_1)

keep_prob = tf.placeholder(tf.float32)
hfc_drop_1 = tf.nn.dropout(hfc_1,keep_prob)

wfc_2 = tf.Variable(tf.truncated_normal([1024,10],stddev=0.1))
bfc_2 = tf.Variable(tf.constant(0.1,shape=[10]))
y_con = tf.nn.softmax(tf.matmul(hfc_drop_1,wfc_2)+bfc_2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_con),reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_con,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = sess.run(accuracy,feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
        print("steps %d,accuracy %g"%(i,train_accuracy))
    sess.run(train_step,feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})
print("test accuracy%g"%accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))
