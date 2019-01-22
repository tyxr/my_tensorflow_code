KERAS_DATASETS_DIR = '/home/xingk/data/'
QUESTION_PAIRS_FILE_URL = 'http://qim.ec.quoracdn.net/quora_duplicate_questions.tsv'
QUESTION_PAIRS_FILE = 'quora_duplicate_questions.tsv'
GLOVE_ZIP_FILE_URL = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'
GLOVE_ZIP_FILE = 'glove.840B.300d.zip'
GLOVE_FILE = 'glove.840B.300d.txt'
Q1_TRAINING_DATA_FILE = './data/q1_train.npy'
Q2_TRAINING_DATA_FILE = './data/q2_train.npy'
LABEL_TRAINING_DATA_FILE = './data/label_train.npy'
WORD_EMBEDDING_MATRIX_FILE = './data/word_embedding_matrix.npy'
NB_WORDS_DATA_FILE = './data/nb_words.json'
MODEL_WEIGHTS_FILE = 'question_pairs_weights.h5'
LOG_PATH = './data/log'
SAVE_PATH = './model/checkpoint'
MAX_NB_WORDS = 200000
TEST_SPLIT = 0.1
RNG_SEED = 13371447


class ModelConfig:
    embedding_dim = 300
    sequence_length = 25
    n_classes = 2
    L2 = 0.1
    hidden_size = 200
    optimizer = 'adam'
    learning_rate = 0.1
    clip_value = 100
    eval_batch = 1000
    dropout_keep_prob = 0.1
    early_stop_learning_rate = 0.00001
    num_epochs = 25
    batch_size = 64
    best_path = './model/bestval'
    save_path = './model/checkpoint'
    tfboard_path = './tensorboard'
    log_path = './data/log'