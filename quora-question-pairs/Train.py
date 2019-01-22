import tensorflow as tf
from Model import SimpleNetwork
import os
import json
import numpy as np
import time
from Process_data import next_batch
from Process_data import print_log
from datetime import datetime
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from Process_data import *
from Config import *
# feed data into feed_dict
def feed_data(q1, q2, y_batch,
              dropout_keep_prob):
    feed_dict = {model.q1: q1,
                 model.q2: q2,
                 model.y: y_batch,
                 model.dropout_keep_prob: dropout_keep_prob}
                 #model.embed_matrix:embeddings}
    return feed_dict

# evaluate current model on devset
def evaluate(sess,Q1_test, Q2_test,  y_test):
    batches = next_batch(Q1_test, Q2_test,y_test)
    data_nums = len(Q1_test)
    total_loss = 0.0
    total_acc = 0.0
    for batch in batches:
        batch_nums = len(batch[0])
        feed_dict = feed_data(*batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        print("acc","batch_nums")
        print(acc,batch_nums)
        total_loss += loss * batch_nums
        total_acc += acc * batch_nums
    print("total_acc  data_nums")
    print(total_acc,data_nums)
    return total_loss / data_nums, total_acc / data_nums

# training
def train():
    # load data
    print_log('Loading training and validation data ...', file=log)
    start_time = time.time()
    #premise_train, premise_mask_train, hypothesis_train, hypothesis_mask_train, y_train = sentence2Index(arg.trainset_path, vocab_dict)
    #premise_dev, premise_mask_dev, hypothesis_dev, hypothesis_mask_dev, y_dev = sentence2Index(arg.devset_path, vocab_dict)
    Q1_train, Q2_train, y_train, Q1_test, Q2_test,  y_test = return_stupid_data()
    print(len(Q1_train), len(Q1_test))
    data_nums = len(Q1_train)
    time_diff = get_time_diff(start_time)
    print_log('Time usage : ', time_diff, file=log)

    # model saving
    saver = tf.train.Saver(max_to_keep=5)
    save_file_dir, save_file_name = os.path.split(config.save_path)
    if not os.path.exists(save_file_dir):
        os.makedirs(save_file_dir)

    # for TensorBoard
    print_log('Configuring TensorBoard and Saver ...', file=log)
    if not os.path.exists(config.tfboard_path):
        os.makedirs(config.tfboard_path)
    tf.summary.scalar('loss', model.loss)
    tf.summary.scalar('accuracy', model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(config.tfboard_path)

    # init
    sess = tf.Session()
    #sess.run(tf.global_variables_initializer(), {model.embed_matrix : embeddings})
    sess.run(tf.global_variables_initializer())

    writer.add_graph(sess.graph)
    # count trainable parameters
    total_parameters = count_parameters()
    print_log('Total trainable parameters : {}'.format(total_parameters), file=log)
    # training
    print_log('Start training and evaluating ...', file=log)
    start_time = time.time()
    total_batch = 0
    best_acc_val = 0.0
    last_improved_batch = 0
    isEarlyStop = False
    last_dev_acc = 0.0
    for epoch in range(num_epochs):
        print_log('Epoch : ', epoch + 1, file=log)
        batches = next_batch(Q1_train,Q2_train,y_train, batchSize=config.batch_size)
        total_loss, total_acc = 0.0, 0.0
        for batch in batches:
            batch_nums = len(batch[0])
            feed_dict = feed_data(*batch, dropout_keep_prob)
            _, batch_loss, batch_acc = sess.run([model.train, model.loss, model.acc], feed_dict=feed_dict)
            total_loss += batch_loss * batch_nums
            total_acc += batch_acc * batch_nums

            # evaluta on devset
            if total_batch % eval_batch == 0:
                # write tensorboard scalar
                s = sess.run(merged_summary, feed_dict=feed_dict)

                writer.add_summary(s, total_batch)

                feed_dict[model.dropout_keep_prob] = 1.0
                loss_val, acc_val = evaluate(sess,Q1_test, Q2_test,  y_test)

                # save model
                saver.save(sess = sess, save_path = save_path + '_dev_loss_{:.4f}.ckpt'.format(loss_val))
                # save best model
                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    last_improved_batch = total_batch
                    saver.save(sess = sess, save_path = best_path)
                    improved_flag = '*'
                else:
                    improved_flag = ''

                # learning rate decay
                if acc_val < last_dev_acc:
                    old_learning_rate = model.learning_rate
                    model.learning_rate *= 0.2
                    print_log('Learning rate change from {0:>6.4} to {1:>6.4} ...'.format(old_learning_rate, model.learning_rate), file = log)

                last_dev_acc = acc_val

                # show batch training information
                time_diff = get_time_diff(start_time)
                msg = 'Epoch : {0:>3}, Batch : {1:>8}, Train Batch Loss : {2:>6.2}, Train Batch Acc : {3:>6.2%}, Dev Loss : {4:>6.2}, Dev Acc : {5:>6.2%}, Time : {6} {7}'
                print_log(msg.format(epoch + 1, total_batch, batch_loss, batch_acc, loss_val, acc_val, time_diff, improved_flag))

            total_batch += 1
            # early stop judge
            if model.learning_rate < early_stop_learning_rate:
                print_log('Learning rate less than threshold, auto-stopping ...', file = log)
                isEarlyStop = True
                break
        if isEarlyStop:
            break

        time_diff = get_time_diff(start_time)
        total_loss, total_acc = total_loss / data_nums, total_acc / data_nums
        msg = '** Epoch : {0:>2} finished, Train Loss : {1:>6.2}, Train Acc : {2:6.2%}, Time : {3}'
        print_log(msg.format(epoch + 1, total_loss, total_acc, time_diff), file = log)

if __name__ == '__main__':
    # read config
    config = ModelConfig

    with open("./data/nb_words.json", 'r') as load_f:
        num_words = json.load(load_f)
        num_words = num_words['nb_words']

    if not os.path.exists(LOG_PATH):
        os.mkdir(LOG_PATH)
    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_path = LOG_PATH+str(dt)
    log = open(log_path, 'w')


    embedding_dim = config.embedding_dim
    sequence_length = config.sequence_length
    n_classes = config.n_classes
    L2 = config.L2
    hidden_size = config.hidden_size
    optimizer = config.optimizer
    learning_rate = config.learning_rate
    clip_value = config.clip_value
    save_path = config.save_path
    eval_batch = config.eval_batch
    dropout_keep_prob = config.dropout_keep_prob
    early_stop_learning_rate = config.early_stop_learning_rate
    best_path = config.best_path
    num_epochs = config.num_epochs

    model = SimpleNetwork(num_words,embedding_dim,sequence_length,\
                          n_classes,L2,hidden_size,optimizer,learning_rate,clip_value)
    train()
