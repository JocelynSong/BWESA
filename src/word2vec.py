import tensorflow as tf
import numpy as np
import logging
import collections
import random

from src.utils import zero_initializer_variable, truncated_normal_initializer_variable
from src.embedding import WordEmbedding

__author__ = "Jocelyn"
logger = logging.getLogger(__name__)


class Word2Vec(object):
    def __init__(self, bwe_config, session, word2id, id2count, id2word, words_to_train, name):
        self._config = bwe_config
        self._session = session
        self.word2id = word2id
        self.id2count = id2count
        self.id2word = id2word
        self.uni_count = [id2count[ind] for ind in range(len(id2count))]
        self.data_index = 0
        self.valid_index = 0
        self.words_to_train = words_to_train
        self.train_words = 0
        self.embedding = WordEmbedding(word2id, self._config.embedding_size, bwe_config.uniform_width, name)
        self.name = name

        self.examples = tf.placeholder(dtype=tf.int64, shape=[self._config.batch_size], name=name+"_examples")
        self.labels = tf.placeholder(dtype=tf.int64, shape=[self._config.batch_size, 1], name=name+"_labels")

        loss_nce = self.nce_loss(self.examples, self.labels)
        tf.summary.scalar("NCE_loss", loss_nce)
        self.loss_nce = loss_nce
        self.loss = loss_nce

        if self._config.normalize:
            embedding_l2_norm = self.embedding.l2_norm
            self.emb_l2_loss = self._config.weight_embedding * embedding_l2_norm
            tf.summary.scalar("emb_l2_loss", embedding_l2_norm)
            self.loss += self.emb_l2_loss
            tf.summary.scalar("emb_total_loss", self.loss)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.train = self.optimize(self.loss)

        self._session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    @property
    def session(self):
        return self._session

    def nce_loss(self, examples, labels):
        """
        :param examples:
        :param labels:
        :return:
        """
        embed = tf.nn.embedding_lookup(self.embedding.W, examples)

        nce_weights = truncated_normal_initializer_variable([len(self.word2id), self._config.embedding_size],
                                                            width=self._config.embedding_size,
                                                            name=self.name + "nce_weights")
        nce_bias = zero_initializer_variable(shape=[len(self.word2id)], name=self.name + "nce_bias")
        # batch monolingual loss
        nce_loss_tensor = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                                        biases=nce_bias,
                                                        labels=labels,
                                                        inputs=embed,
                                                        num_sampled=self._config.num_negative_samples,
                                                        num_classes=len(self.word2id)))
        return nce_loss_tensor

    def optimize(self, loss):
        lr = self._config.optimizer_config.lr * tf.maximum(0.0001, 1.0 - float(self.train_words) / self.words_to_train)
        self.lr = lr
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        train = optimizer.minimize(loss, global_step=self.global_step, gate_gradients=optimizer.GATE_NONE)
        return train

    def train_step(self, this_examples, this_labels):
        input_feed = {self.examples: this_examples,
                      self.labels: this_labels}
        output_feed = [self.train, self.loss]

        _, loss = self._session.run(output_feed, input_feed)
        return loss

    def predict_step(self, this_examples, this_labels):
        input_feed = {self.examples: this_examples,
                      self.labels: this_labels}
        output_feed = [self.loss]

        loss = self._session.run(output_feed, input_feed)
        return loss

    def get_batch_valid(self, data):
        batch_size = self._config.batch_size
        num_skips = self._config.num_skips
        skip_window = self._config.skip_window

        batch = np.ndarray(shape=(batch_size,), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1
        buffer = collections.deque(maxlen=span)

        if self.valid_index + span > len(data):
            self.valid_index = 0
        buffer.extend(data[self.valid_index: self.valid_index + span])
        self.valid_index += span

        for i in range(batch_size // num_skips):
            target = skip_window
            targets_to_avoids = [skip_window]
            for j in range(num_skips):
                while target in targets_to_avoids:
                    target = random.randint(0, span - 1)
                targets_to_avoids.append(target)
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[target]
            if self.valid_index == len(data):
                buffer.extend(data[: span])
                self.valid_index = span
            else:
                buffer.append(data[self.valid_index])
                self.valid_index += 1
        self.valid_index = (self.valid_index + len(data) - span) % len(data)
        return batch, labels

    def get_batch(self, data, valid=False):
        if valid:
            return self.get_batch_valid(data)

        self.train_words += 1

        batch_size = self._config.batch_size
        num_skips = self._config.num_skips
        skip_window = self._config.skip_window

        batch = np.ndarray(shape=(batch_size, ), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1
        buffer = collections.deque(maxlen=span)

        if self.data_index + span > len(data):
            self.data_index = 0
        buffer.extend(data[self.data_index: self.data_index + span])
        self.data_index += span

        for i in range(batch_size // num_skips):
            target = skip_window
            targets_to_avoids = [skip_window]
            for j in range(num_skips):
                while target in targets_to_avoids:
                    target = random.randint(0, span - 1)
                targets_to_avoids.append(target)
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[target]
            if self.data_index == len(data):
                buffer.extend(data[: span])
                self.data_index = span
            else:
                buffer.append(data[self.data_index])
                self.data_index += 1
        self.data_index = (self.data_index + len(data) - span) % len(data)
        return batch, labels

    def save(self, save_file):
        f = open(save_file, "w", encoding="utf-8")
        result = self._session.run([self.embedding.W])
        embedding = result[0]
        for i in range(len(self.id2word)):
            f.write(self.id2word[i])
            for data in embedding[i]:
                f.write(" ")
                f.write(str(data))
            f.write("\n")
        f.close()











































