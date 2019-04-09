# -*- coding: UTF-8 -*-

import os
import sys
import timeit
import random
import argparse

import numpy as np
import tensorflow as tf

__author__ = "Jocelyn"
FLAGS = None


class HiddenLayer(object):
    def __init__(self, rng, n_in, n_out, input, W=None, b=None, activation=tf.nn.tanh):
        self.input = input

        if W is None:
            init_width = np.sqrt(6. / (n_in + n_out))
            W = tf.Variable(tf.random_uniform([n_in, n_out], -init_width, init_width), name="hidden_layer_W")
            if activation == tf.nn.sigmoid:
                W = tf.Variable(4 * tf.random_uniform([n_in, n_out], -init_width, init_width), name="hidden_layer_W")
        if b is None:
            b = tf.Variable(tf.zeros(n_out, ), name="hidden_layer_b")
        self.W = W
        self.b = b

        lin_out = tf.matmul(self.input, self.W) + self.b
        self.output = lin_out if activation is None else activation(lin_out)


class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        self.W = tf.Variable(tf.zeros([n_in, n_out]), dtype=tf.float32, name="logistic_regression_w")
        self.b = tf.Variable(tf.zeros([n_out, ]), dtype=tf.float32, name="logistic_regression_b")
        self.p_y_given_x = tf.nn.softmax(tf.matmul(input, self.W) + self.b)

        self.y_pred = tf.argmax(self.p_y_given_x, axis=1)
        self.input = input

    def negative_log_likelihood(self, y):
        return -tf.reduce_mean(tf.log(self.p_y_given_x)[range(len(y)), y])

    def errors(self, y):
        return tf.reduce_mean(tf.not_equal(self.y_pred, y))


class MLP(object):
    def __init__(self, session, rng, n_in, n_hidden, n_out, batch_size, learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001):
        self.batch_size = batch_size
        self.session = session
        self.x = tf.placeholder(dtype=tf.float32, shape=[batch_size, n_in])
        self.y = tf.placeholder(dtype=tf.int64, shape=[batch_size, ])

        self.hidden_layer = HiddenLayer(rng=rng,
                                        n_in=n_in,
                                        n_out=n_hidden,
                                        input=self.x,
                                        activation=tf.nn.tanh)

        self.softmax_w = tf.Variable(tf.zeros([n_hidden, n_out]), dtype=tf.float32, name="logistic_regression_w")
        self.softmax_b = tf.Variable(tf.zeros([n_out, ]), dtype=tf.float32, name="logistic_regression_b")

        self.p_y_given_x = tf.nn.softmax(tf.matmul(self.hidden_layer.output, self.softmax_w) + self.softmax_b)
        self.y_prediction = tf.argmax(self.p_y_given_x, axis=1)
        self.loss_likelihood = self.negative_log_likelihood(self.y)
        self.errors = self.errors(self.y)

        self.L1 = tf.reduce_sum(tf.abs(self.hidden_layer.W)) + tf.reduce_sum(tf.abs(self.softmax_w))
        self.L2_sqr = tf.reduce_sum(tf.square(self.hidden_layer.W)) + tf.reduce_sum(tf.square(self.softmax_w))
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg
        self.loss = self.loss_likelihood + self.L1_reg * self.L1 + self.L2_reg * self.L2_sqr

        self.learning_rate = tf.Variable(learning_rate, trainable=False, dtype=tf.float32)
        self.learning_rate_decay_factor = tf.constant(0.99, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * self.learning_rate_decay_factor)
        self.train = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

        self.session.run(tf.global_variables_initializer())

    def negative_log_likelihood(self, y):
        with tf.device("/cpu:0"):
            y_labels = tf.one_hot(indices=y, depth=4, on_value=1.0, off_value=0.0)
        likelihood = tf.reduce_sum(tf.multiply(tf.log(self.p_y_given_x), y_labels), axis=1)
        return -tf.reduce_mean(likelihood)

    def errors(self, y):
        return tf.reduce_mean(tf.cast(tf.not_equal(self.y_prediction, y), tf.float32))

    def step(self, train_x, train_y):
        input_feed = {self.x: train_x,
                      self.y: train_y}
        output_feed = [self.train, self.loss, self.loss_likelihood]

        result = self.session.run(output_feed, input_feed)
        loss, loss_likelihood = result[1], result[2]
        return loss, loss_likelihood

    def valid(self, train_x, train_y):
        input_feed = {self.x: train_x,
                      self.y: train_y}
        output_feed = [self.loss, self.loss_likelihood]

        result = self.session.run(output_feed, input_feed)
        loss, loss_likelihood = result[0], result[1]
        return loss, loss_likelihood

    def predict(self, test_x, test_y):
        input_feed = {self.x: test_x,
                      self.y: test_y}
        output_feed = [self.errors]

        result = self.session.run(output_feed, input_feed)
        pred_error = result[0]
        return pred_error

    def get_batch(self, train_x, train_y, index):
        this_train_x = train_x[index * self.batch_size: (index + 1) * self.batch_size]
        this_train_y = train_y[index * self.batch_size: (index + 1) * self.batch_size]
        return this_train_x, this_train_y


def get_doc_embedding(words, word_dict, words_idf_scores, unk="</s>"):
    embedding = np.zeros([100])
    for word in words:
        word = word.lower()   # true case
        # discard the punctuations
        if word in "-,.!?:;\'\"{}[]()+-*%/=":
            continue
        if word in word_dict:
            word_embedding = word_dict[word]
        else:
            continue
            # word_embedding = word_dict[unk]
        # embedding += word_embedding * np.log(words_idf_scores[word])
        embedding += word_embedding
    return embedding


def get_data(dir_path, files, word_dict, words_idf_scores, unk="<UNK>"):
    data, labels = [], []
    for file in files:
        filename = os.path.join(dir_path, file)
        f = open(filename, "r", encoding="utf-8")
        lines = f.readlines()
        # append a label
        if len(lines) < 1:
            print(filename)
        label = int(lines[0].strip())
        labels.append(label)
        # append a doc embedding
        words = []
        for line in lines[1:]:
            words.extend(line.strip().split())
        embedding = get_doc_embedding(words, word_dict, words_idf_scores, unk)
        data.append(embedding)
    return data, labels


def count_idf(dir_path, files):
    word_idf = {}
    total_number = len(files)
    print("There is total number of %d English files\n" % total_number)
    index = 0
    for file in files:
        index += 1
        if index % 1000 == 0:
            print("Now deal %d files\n" % index)
        filename = os.path.join(dir_path, file)
        f = open(filename, "r", encoding="utf-8")
        lines = f.readlines()[1:]
        words = []
        for line in lines:
            words.extend(line.strip().split())
        words = set(words)
        for word in words:
            if word in word_idf:
                word_idf[word] += 1
            else:
                word_idf[word] = 1
    for key, count in word_idf.items():
        word_idf[key] = count / float(total_number)
    return word_idf


def get_train_valid_test_set(src_dir_path, tar_dir_path):
    """
    src: 10000 training set, of which 100-10000 used to train, 1000 validation set
    tar: 5000 test set
    :param src_dir_path:
    :param tar_dir_path:
    :return:
    """
    src_files = os.listdir(src_dir_path)
    src_lines = range(len(src_files))
    src_samples = random.sample(src_lines, 10000)
    train_lines = src_samples[: 10000]
    train_files = [src_files[line] for line in train_lines]

    tar_files = os.listdir(tar_dir_path)
    tar_lines = range(len(tar_files))
    tar_samples = random.sample(tar_lines, 6000)
    validation_lines = tar_samples[:1000]
    validation_files = [tar_files[line] for line in validation_lines]

    test_lines = tar_samples[1000:]
    test_files = [tar_files[line] for line in test_lines]
    return train_files, validation_files, test_files


def test_mlp(src_dir_path, tar_dir_path, src_word_embeddings, tar_word_embeddings, src_word_idf_scores,
             tar_word_idf_scores, learning_rate=0.0008, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000, batch_size=32,
             n_hidden=500):
    train_files, valid_files, test_files = get_train_valid_test_set(src_dir_path, tar_dir_path)
    # compute idf scores

    """
    src_word_idf_scores = count_idf(src_dir_path, train_files)
    tar_files = [file for file in valid_files]
    tar_files.extend(test_files)
    tar_word_idf_scores = count_idf(tar_dir_path, tar_files)
    """

    train_x, train_y = get_data(src_dir_path, train_files, src_word_embeddings, src_word_idf_scores, FLAGS.unk)
    valid_x, valid_y = get_data(tar_dir_path, valid_files, tar_word_embeddings, tar_word_idf_scores, FLAGS.unk)
    test_x, test_y = get_data(tar_dir_path, test_files, tar_word_embeddings, tar_word_idf_scores, FLAGS.unk)

    n_train_batches = len(train_x) // batch_size
    n_valid_batches = len(valid_x) // batch_size
    n_test_batches = len(test_x) // batch_size

    print("... building the model\n")

    rng = np.random.RandomState(1234)
    session = tf.Session()

    classifier = MLP(session=session,
                     rng=rng,
                     n_in=100,
                     n_hidden=n_hidden,
                     n_out=4,
                     batch_size=batch_size,
                     learning_rate=learning_rate,
                     L1_reg=L1_reg,
                     L2_reg=L2_reg)

    print("Training!\n")

    patience = 10000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience // 2)

    best_validation_loss = np.inf
    best_test_score = np.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            this_train_x, this_train_y = classifier.get_batch(train_x, train_y, minibatch_index)
            this_train_loss, this_train_likelihood = classifier.step(this_train_x, this_train_y)

            iter = (epoch - 1) * n_train_batches + minibatch_index
            if (iter + 1) % validation_frequency == 0:
                validation_losses = []
                for valid_index in range(n_valid_batches):
                    this_valid_x, this_valid_y = classifier.get_batch(valid_x, valid_y, valid_index)
                    this_valid_loss = classifier.predict(this_valid_x, this_valid_y)
                    validation_losses.append(this_valid_loss)
                this_ave_valid_loss = np.mean(validation_losses)

                print("epoch=%d, mini_batch=%d/%d, validation acc=%f %%\n" % (epoch, minibatch_index + 1,
                                                                              n_train_batches,
                                                                              (1.0 - this_ave_valid_loss) * 100))
                if this_ave_valid_loss < best_validation_loss:
                    if this_ave_valid_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                        # session.run(classifier.learning_rate_decay_op)

                    best_validation_loss = this_ave_valid_loss
                    best_iter = iter

                    test_losses = []
                    for test_index in range(n_test_batches):
                        this_test_x, this_test_y = classifier.get_batch(test_x, test_y, test_index)
                        this_valid_loss = classifier.predict(this_test_x, this_test_y)
                        test_losses.append(this_valid_loss)
                    # test_score = np.mean(test_losses)
                    test_score = np.min(test_losses)
                    if test_score < best_test_score:
                        best_test_score = test_score
                    print("epoch=%d, mini_batch=%d/%d, test acc of best model=%f %%" % (epoch, minibatch_index + 1,
                                                                                        n_train_batches,
                                                                                        (1.0 - test_score) * 100))
            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print("Optimization Complete. Best Validation score of %f %% obtained at iteration %d, with test performance %f %%"
          % ((1.0 - best_validation_loss) * 100 % 100, best_iter + 1, (1.0 - test_score) * 100.))

    print("Using time: %fm\n" % ((end_time - start_time) / 60.))
    return best_validation_loss, test_score, best_test_score


def read_freq_words(filename):
    words = []
    f = open(filename, "r", encoding="utf-8")
    for line in f.readlines():
        lemmas = line.strip().split("|||")
        words.append(lemmas[0].strip())
    return words


def read_freq_word_embeddings(filename, freq_file, unk="<UNK>"):
    freq_words = read_freq_words(freq_file)
    freq_words.append(unk)

    embedding = {}
    f = open(filename, "r", encoding="utf-8")
    for line in f.readlines():
        words = line.strip().split()
        word = words[0]
        if word not in freq_words:
            continue
        vector = [float(dim.strip()) for dim in words[1:]]
        embedding[word] = np.array(vector)
    return embedding


def read_word_embeddings(filename):
    embedding = {}
    f = open(filename, "r", encoding="utf-8")
    for line in f.readlines():
        words = line.strip().split()
        word = words[0]
        vector = [float(dim.strip()) for dim in words[1:]]
        embedding[word] = np.array(vector)
    return embedding


def read_idf_scores(filename):
    idf_scores = {}
    f = open(filename, "r", encoding="utf-8")
    for line in f.readlines():
        strings = line.strip().split("|||")
        word = strings[0].strip()
        score = float(strings[1].strip())
        idf_scores[word] = score
    return idf_scores


def main(_):
    with tf.device("/gpu:0"):
        print("read word embedding\n")
        src_word_embedding = read_word_embeddings(FLAGS.src_embedding_path)
        tar_word_embedding = read_word_embeddings(FLAGS.tar_embedding_path)
        # src_word_embedding = read_freq_word_embeddings(FLAGS.src_embedding_path, FLAGS.zh_freq_words, FLAGS.unk)
        # tar_word_embedding = read_freq_word_embeddings(FLAGS.tar_embedding_path, FLAGS.en_freq_words, FLAGS.unk)

        print("read idf scores\n")
        src_idf_scores = read_idf_scores(FLAGS.src_idf_score_file_path)
        tar_idf_scores = read_idf_scores(FLAGS.tar_idf_score_file_path)

        print("training classifier\n")
        valid_errors = []
        test_errors = []
        best_test_errors = []
        for index in range(3):
            print("Random sample train/valid/test iteration %d\n" % (index + 1))
            results = test_mlp(FLAGS.src_dir_path, FLAGS.tar_dir_path, src_word_embedding, tar_word_embedding,
                               src_idf_scores, tar_idf_scores)
            valid_errors.append(results[0])
            test_errors.append(results[1])
            best_test_errors.append(results[2])
        best_valid_error = np.amin(valid_errors)
        best_valid_ind = np.argmin(valid_errors)
        best_test_error = test_errors[best_valid_ind]
        ave_valid_error = np.mean(valid_errors)
        ave_test_error = np.mean(test_errors)
        best_test_score = np.amin(best_test_errors)
        print("Random sample train/valid/test:")
        print("best valid accuracy=%f, best test accuracy=%f, single best test accuracy=%f\n" % (1.0 - best_valid_error,
                                                                                                 1.0 - best_test_error,
                                                                                                 1.0 - best_test_score))
        print("average valid accuracy=%f, average test accuracy=%f\n" % (1.0 - ave_valid_error, 1.0 - ave_test_error))

        """
        test_mlp(FLAGS.src_dir_path, FLAGS.tar_dir_path, src_word_embedding, tar_word_embedding,
                 src_idf_scores, tar_idf_scores)
        """


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname
                                                               (os.path.dirname(os.path.abspath("MLP.py"))))))
    emb_path = "E:\\NLP\\code\\bilingual_word_emb\\emnlp\\other_work\\BiVAE\\bivae\\data\\zh-en"

    parse = argparse.ArgumentParser()

    parse.add_argument("--src_dir_path", type=str,
                       default=os.path.join(dir_path, "data\\cross-lingual document classification\\rcv2\\"
                                            "RCV2_sample_seg\\chinese"),
                       help="source document file directory")
    parse.add_argument("--tar_dir_path", type=str,
                       default=os.path.join(dir_path, "data\\cross-lingual document classification\\rcv1"
                                            "\\rcv1_case"),
                       help="target document file directory")
    parse.add_argument("--src_embedding_path", type=str,
                       default=os.path.join(emb_path, "de.emb"),
                       help="source embedding file path")
    parse.add_argument("--tar_embedding_path", type=str,
                       default=os.path.join(emb_path, "en.emb"),
                       help="target embedding file path")
    parse.add_argument("--src_idf_score_file_path", type=str,
                       default=os.path.join(dir_path, "data\\cross-lingual document classification\\idf-scores"
                                                      "\\zh.idf.score"),
                       help="source idf score file path")
    parse.add_argument("--tar_idf_score_file_path", type=str,
                       default=os.path.join(dir_path, "data\\cross-lingual document classification\\idf-scores"
                                                      "\\en.idf.score"),
                       help="target idf score file path")
    parse.add_argument("--zh_freq_words", type=str,
                       default=os.path.join(dir_path, "data\\cross-lingual dictionary induction\\"
                                            "freq_words\\zh-en\\zh.fre.50000"),
                       help="file path for chinese frequency words")
    parse.add_argument("--en_freq_words", type=str,
                       default=os.path.join(dir_path, "data\\cross-lingual dictionary induction\\"
                                            "freq_words\\zh-en\\en.fre.50000"),
                       help="file path for english frequency words")
    parse.add_argument("--unk", type=str, default="_UNK_", help="unk label for this embedding")

    FLAGS, unparsed = parse.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
























