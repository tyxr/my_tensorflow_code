from __future__ import print_function
import numpy as np
import csv, json
from zipfile import ZipFile
from os.path import expanduser, exists
from os import mkdir
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.data_utils import get_file
from Config import *
import time
from datetime import timedelta
from sklearn.model_selection import train_test_split
import tensorflow as tf

def process_data():
    if not exists('./data'):
        mkdir('./data')

    # Else download and extract questions pairs data
    if not exists(KERAS_DATASETS_DIR + QUESTION_PAIRS_FILE):
        get_file(QUESTION_PAIRS_FILE, QUESTION_PAIRS_FILE_URL)

    print("Processing", QUESTION_PAIRS_FILE)

    question1 = []
    question2 = []
    is_duplicate = []
    with open(KERAS_DATASETS_DIR + QUESTION_PAIRS_FILE, encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            question1.append(row['question1'])
            question2.append(row['question2'])
            is_duplicate.append(row['is_duplicate'])

    print('Question pairs: %d' % len(question1))

    # Build tokenized word index
    questions = question1 + question2
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(questions)
    question1_word_sequences = tokenizer.texts_to_sequences(question1)
    question2_word_sequences = tokenizer.texts_to_sequences(question2)
    word_index = tokenizer.word_index

    print("Words in index: %d" % len(word_index))

    # Download and process GloVe embeddings
    if not exists(KERAS_DATASETS_DIR + GLOVE_ZIP_FILE):
        zipfile = ZipFile(get_file(GLOVE_ZIP_FILE, GLOVE_ZIP_FILE_URL))
        zipfile.extract(GLOVE_FILE, path=KERAS_DATASETS_DIR)

    print("Processing", GLOVE_FILE)

    embeddings_index = {}
    with open(KERAS_DATASETS_DIR + GLOVE_FILE, encoding='utf-8') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = embedding

    print('Word embeddings: %d' % len(embeddings_index))

    # Prepare word embedding matrix
    nb_words = min(MAX_NB_WORDS, len(word_index))
    word_embedding_matrix = np.zeros((nb_words + 1, config.embedding_dim))
    for word, i in word_index.items():
        if i > MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            word_embedding_matrix[i] = embedding_vector

    print('Null word embeddings: %d' % np.sum(np.sum(word_embedding_matrix, axis=1) == 0))

    # Prepare training data tensors
    q1_data = pad_sequences(question1_word_sequences, maxlen=config.sequence_length)
    q2_data = pad_sequences(question2_word_sequences, maxlen=config.sequence_length)
    labels = np.array(is_duplicate, dtype=int)
    print('Shape of question1 data tensor:', q1_data.shape)
    print('Shape of question2 data tensor:', q2_data.shape)
    print('Shape of label tensor:', labels.shape)

    # Persist training and configuration data to files
    np.save(open(Q1_TRAINING_DATA_FILE, 'wb'), q1_data)
    np.save(open(Q2_TRAINING_DATA_FILE, 'wb'), q2_data)
    np.save(open(LABEL_TRAINING_DATA_FILE, 'wb'), labels)
    np.save(open(WORD_EMBEDDING_MATRIX_FILE, 'wb'), word_embedding_matrix)
    with open(NB_WORDS_DATA_FILE, 'w') as f:
        json.dump({'nb_words': nb_words}, f)

def next_batch(Q1_data, Q2_data,  y, batchSize = 64, shuffle = True):
    """
    :param premise_mask: actual length of premise
    :param hypothesis_mask: actual length of hypothesis
    :param shuffle: boolean, shuffle dataset or not
    :return: generate a batch of data (premise, premise_mask, hypothesis, hypothesis_mask, label)
    """
    sampleNums = len(Q1_data)
    batchNums = int((sampleNums - 1) / batchSize) + 1

    if shuffle:
        indices = np.random.permutation(np.arange(sampleNums))
        premise = Q1_data[indices]
        hypothesis = Q2_data[indices]
        y = y[indices]

    for i in range(batchNums):
        startIndex = i * batchSize
        endIndex = min((i + 1) * batchSize, sampleNums)
        yield (Q1_data[startIndex : endIndex],
               Q2_data[startIndex : endIndex],
               y[startIndex : endIndex])
def print_log(*args, **kwargs):
    print(*args)
    if len(kwargs) > 0:
        print(*args, **kwargs)
    return None
def get_time_diff(startTime):
    endTime = time.time()
    diff = endTime - startTime
    return timedelta(seconds = int(round(diff)))
def return_stupid_data():
    q1_data = np.load(Q1_TRAINING_DATA_FILE)
    q2_data = np.load(Q2_TRAINING_DATA_FILE)
    X = np.stack((q1_data, q2_data), axis=1)
    labels = np.load(LABEL_TRAINING_DATA_FILE)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=TEST_SPLIT, random_state=RNG_SEED)
    Q1_train = X_train[:, 0]
    Q2_train = X_train[:, 1]
    Q1_test = X_test[:, 0]
    Q2_test = X_test[:, 1]

    return Q1_train,Q2_train,y_train,Q1_test,Q2_test, y_test
def count_parameters():
    totalParams = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variableParams = 1
        for dim in shape:
            variableParams *= dim.value
        totalParams += variableParams
    return totalParams
def print_shape(varname, var):
    """
    :param varname: tensor name
    :param var: tensor variable
    """
    print('{0} : {1}'.format(varname, var.get_shape()))
if __name__=='__main__':
    config = ModelConfig
    process_data()


