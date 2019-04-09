import random
import math

import tensorflow as tf
import numpy as np
import logging
import matplotlib.pyplot as plt

__author__ = "Jocelyn"


def uniform_initializer_variable(init_width, shape, name):
    var = tf.Variable(tf.random_uniform(shape, -init_width, init_width), name=name)
    return var


def truncated_normal_initializer_variable(shape, width, name=""):
    var = tf.Variable(tf.truncated_normal(shape=shape, stddev=1.0/math.sqrt(width)), name=name)
    return var


def zero_initializer_variable(shape, name):
    var = tf.Variable(tf.zeros(shape), name=name)
    return var


def matrix_initialization_variable(w, name):
    var = tf.Variable(w, name=name)
    return var


def get_sen_mask(batch_size, max_len, sen_len):
    mask = np.zeros([batch_size, max_len])
    for i in range(batch_size):
        if sen_len[i] > max_len:
            mask[i, :] = 1
        else:
            mask[i, :sen_len[i]] = 1
    return mask


def pre_logger(log_file_name, file_handler_level=logging.DEBUG, screen_handler_level=logging.INFO):
    """
    define log format
    :param log_file_name:
    :param file_handler_level:
    :param screen_handler_level:
    :return:
    """
    log_formatter = logging.Formatter(fmt='%(asctime)s [%(processName)s, %(process)s] [%(levelname)-5.5s]  %(message)s',
                                      datefmt='%m-%d %H:%M')
    init_logger = logging.getLogger(log_file_name)
    init_logger.setLevel(logging.DEBUG)

    # file handler
    file_handler = logging.FileHandler("log/{}.log".format(log_file_name))
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(file_handler_level)

    # screen handler
    screen_handler = logging.StreamHandler()
    screen_handler.setLevel(screen_handler_level)

    init_logger.addHandler(file_handler)
    init_logger.addHandler(screen_handler)
    return init_logger


def visualize(train_loss, valid_loss):
    time = range(len(train_loss))

    plt.plot(time, train_loss, label='train loss')
    plt.scatter(time, train_loss)
    plt.plot(time, valid_loss, label='validation loss')
    plt.scatter(time, valid_loss)

    plt.xlabel('training epoch')
    plt.ylabel('loss')
    plt.title('Loss Value Variation')
    plt.legend()
    plt.show()


def visualize_training(train_loss):
    time = range(len(train_loss))

    plt.plot(time, train_loss, label='train loss')
    plt.scatter(time, train_loss)

    plt.xlabel('training epoch')
    plt.ylabel('loss')
    plt.title('Loss Value Variation')
    plt.legend()
    plt.show()


def get_log_name(config):
    log_name = "embedding_size%d_weight_embedding%f_alpha%f_beta%f" % (config.embedding_size, config.weight_embedding,
                                                                       config.alpha, config.beta)
    return log_name


def get_lines(filename):
    f = open(filename, "r", encoding="utf-8")
    lines = f.readlines()
    f.close()
    return lines


def output_sentence(id2word, sentences):
    for sen in sentences:
        s = ""
        for word in sen:
            s += id2word[word]+" "
        print(s)
        print("\n")


def sample_indexes(batch_size, max_len, top_k):
    """
    tf.gather
    :param batch_size:
    :param max_len:
    :param top_k:
    :return:
    """
    samples = []
    for i in range(batch_size):
        sample = []
        for j in range(max_len):
            index = random.choice(range(top_k))
            new_index = [i, j, index]
            sample.append(new_index)
        samples.append(sample)
    return samples


def expand_gather_dimension(matrix):
    indexes = []
    for i in range(len(matrix)):
        alignments = []
        for j in range(matrix[i]):
            alignments.append([i, matrix[i][j]])
        indexes.append(alignments)
    return indexes


def read_stop_words(filename):
    f = open(filename, "r", encoding="utf-8")
    stop_words = []
    for line in f.readlines():
        stop_words.append(line.strip())
    return stop_words


def read_stop_words_lower(filename):
    f = open(filename, "r", encoding="utf-8")
    stop_words = []
    for line in f.readlines():
        stop_words.append(line.strip().lower())
    return stop_words




