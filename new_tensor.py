import tensorflow as tf
from stupid_feature import read_feature
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
INPUT = 10
OUTPUT = 1
H1 = 5
TRAINING_STEPS = 10000


def train(x_train,y_train,x_test,y_test):

    sess = tf.Session()

    seed = 2
    tf.set_random_seed(seed)
    np.random.seed(seed)

    x = tf.placeholder(tf.float32,[None,INPUT])
    y = tf.placeholder(tf.float32,[None,OUTPUT])

    
    w1 = tf.Variable(tf.random_normal([INPUT,H1]))
    b1 = tf.Variable(tf.zeros([H1]))
    w2 = tf.Variable(tf.random_normal([H1,1]))
    b2 = tf.Variable(tf.zeros([1]))

    hidden_output = tf.nn.relu(tf.add(tf.matmul(x,w1),b1))
    final_output = tf.nn.relu(tf.add(tf.matmul(hidden_output,w2),b2))

    loss = tf.reduce_mean(tf.square(y - final_output))

    my_opt = tf.train.GradientDescentOptimizer(0.005)
    train_step = my_opt.minimize(loss)
    init = tf.global_variables_initializer()
    sess.run(init)


    loss_vec = []
    test_loss = []
    for i in range(500):

        _,temp_loss=sess.run([train_step,loss],feed_dict={x:x_train,y:y_train})

        test_temp_loss=sess.run(loss,feed_dict={x:x_test,y:y_test})

        #temp_loss = sess.run(loss,feed_dict={x:x_train,y:y_train})
        #sess.run(train_step,feed_dict={x_data:rand_x,y_target:rand_y})

        loss_vec.append(np.sqrt(temp_loss))
        #test_temp_loss=sess.run(loss,feed_dict={x_data:x_vals_test,y_target:np.transpose([y_vals_test])})

        test_loss.append(np.sqrt(test_temp_loss))
        if (i+1)%50 == 0:
            print('Generation:'+str(i+1)+'.Loss='+str(temp_loss))

    #print(test_loss)
    plt.plot(loss_vec,'k-',label='Train Loss')
    plt.plot(test_loss,'r--',label='Test Loss')
    plt.title('Loss (MSE) per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()
        

def _main():
    X,Y_raw = read_feature("ALL3.txt")
    encoder = LabelEncoder()
    encoder.fit(Y_raw)
    Y = encoder.transform(Y_raw).reshape((len(Y_raw), 1))
    train(X[:100],Y[:100],X[100:],Y[100:])
     

if __name__ == '__main__':
    _main()

