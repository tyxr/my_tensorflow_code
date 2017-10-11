import tensorflow as tf
from stupid_feature import read_feature
from sklearn.preprocessing import LabelEncoder

INPUT = 10
OUTPUT = 1
H1 = 5
TRAINING_STEPS = 10000


def train(x_train,y_train,x_test,y_test):
    w1 = tf.Variable(tf.truncated_normal([INPUT,H1],stddev = 0.1))
    b1 = tf.Variable(tf.zeros([H1]))
    w2 = tf.Variable(tf.zeros([H1,1]))
    b2 = tf.Variable(tf.zeros([1]))
    x = tf.placeholder(tf.float32,[None,INPUT])
    y = tf.placeholder(tf.float32,[None,OUTPUT])
    hidden1 = tf.nn.relu(tf.matmul(x,w1)+b1)
    y_pre = tf.sigmoid(tf.matmul(hidden1,w2)+b2)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y,labels=y_pre)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean

    learning_rate = 0.1
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    correct_prediction = tf.equal(y,y_pre)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            
            #_, loss_value, step = sess.run(train_step, feed_dict={x: x_train, y: y_train})
            sess.run(train_step, feed_dict={x: x_train, y: y_train})
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (i)
                #saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
        test_acc = sess.run(accuracy,feed_dict={x:x_test, y:y_test})
        print(test_acc)

def main(argv=None):
    X,Y_raw = read_feature("ALL3.txt")
    encoder = LabelEncoder()
    encoder.fit(Y_raw)
    Y = encoder.transform(Y_raw).reshape((len(Y_raw), 1))
    train(X[:100],Y[:100],X[100:],Y[100:])
     

if __name__ == '__main__':
    main(argv=None)

