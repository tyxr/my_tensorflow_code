import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, DropoutWrapper
from tensorflow.keras.layers import Embedding,TimeDistributed,Dense,Lambda
from Process_data import print_shape
import numpy as np
class SimpleNetwork():
    def __init__(self,num_words,embedding_dim,sequence_length,n_classes,L2,hidden_size,optimizer,learning_rate,clip_value):
        #para init
        self.num_words = num_words
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.n_classes = n_classes
        self.L2 = L2
        self.hidden_size = hidden_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.clip_value = clip_value
        self._placeholder_init()

        # model operation
        self.logits = self._logits_op()
        self.loss = self._loss_op()
        self.acc = self._acc_op()
        self.train = self._training_op()
        tf.add_to_collection('train_mini', self.train)


    def _placeholder_init(self):
        self.q1 = tf.placeholder(tf.float32, [None, self.sequence_length], 'q1')
        self.q2 = tf.placeholder(tf.float32, [None, self.sequence_length], 'q2')
        self.y = tf.placeholder(tf.float32, None, 'y_true')#这里的1本身应该是类别，我们二分类就输出一个就好了。
        #self.embed_matrix = tf.placeholder(tf.float32, [self.num_words+1, self.embedding_dim], 'embed_matrix')
        self.embed_matrix = np.load('./data/word_embedding_matrix.npy')

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
    def _logits_op(self):
        print(self.q1)
        print(self.embed_matrix)
        print(self.num_words+1)
        q1 = Embedding(self.num_words+1, self.embedding_dim,weights=[self.embed_matrix],input_length=self.sequence_length,trainable=False )(self.q1)
        q1 = TimeDistributed(Dense(self.embedding_dim, activation='relu'))(q1)
        q1 = Lambda(lambda x: tf.reduce_max(x, axis=1), output_shape=(self.embedding_dim,))(q1)

        q2 = Embedding(self.num_words+1, self.embedding_dim,
                           weights=[self.embed_matrix],input_length=self.sequence_length,
                           trainable=False )(self.q2)
        q2 = TimeDistributed(Dense(self.embedding_dim, activation='relu'))(q2)
        q2 = Lambda(lambda x: tf.reduce_max(x, axis=1), output_shape=(self.embedding_dim,))(q2)
        features = tf.concat([q1, q2], axis=1)
        logits = self._feedForwardBlock(features, self.hidden_size, self.n_classes, 'feed_forward')

        return logits

    def _loss_op(self):
        with tf.name_scope('cost'):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.logits)
            loss = tf.reduce_mean(losses, name='loss_val')
            weights = [v for v in tf.trainable_variables() if ('w' in v.name) or ('kernel' in v.name)]
            l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights]) * self.L2
            loss += l2_loss
        return loss
    def _acc_op(self):
        with tf.name_scope('acc'):
            label_pred = self.logits
            label_true = self.y
            correct_pred = tf.equal(tf.round(label_pred), tf.round(label_true))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='Accuracy')
        return accuracy

    def _feedForwardBlock(self, inputs, hidden_dims, num_units, scope):
        """
        :param inputs: tensor with shape (batch_size, 4 * 2 * hidden_size)
        :param scope: scope name
        :return: output: tensor with shape (batch_size, num_units)
        """
        with tf.variable_scope(scope):

            initializer = tf.random_normal_initializer(0.0, 0.1)

            with tf.variable_scope('feed_foward_layer1'):
                inputs = tf.nn.dropout(inputs, self.dropout_keep_prob)
                outputs = tf.layers.dense(inputs, hidden_dims, tf.nn.relu, kernel_initializer = initializer)
            with tf.variable_scope('feed_foward_layer2'):
                outputs = tf.nn.dropout(outputs, self.dropout_keep_prob)
                results = tf.layers.dense(outputs, 1, tf.nn.sigmoid, kernel_initializer = initializer)
                return results
    def _training_op(self):
        with tf.name_scope('training'):
            if self.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
            elif self.optimizer == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            elif self.optimizer == 'momentum':
                optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9)
            elif self.optimizer == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            elif self.optimizer == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
            elif self.optimizer == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            else:
                ValueError('Unknown optimizer : {0}'.format(self.optimizer))
        gradients, v = zip(*optimizer.compute_gradients(self.loss))
        if self.clip_value is not None:
            gradients, _ = tf.clip_by_global_norm(gradients, self.clip_value)
        train_op = optimizer.apply_gradients(zip(gradients, v))
        return train_op