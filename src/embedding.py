import tensorflow as tf
import logging

from src.utils import uniform_initializer_variable, matrix_initialization_variable

__author__ = "Jocelyn"
logger = logging.getLogger(__name__)


class Embedding(object):
    def __init__(self, w=None, size=10000, dimension=50, init_width=1.0, name="embedding"):
        if w is None:
            self.size = size
            self.dim = dimension
            init_width = init_width / self.dim
            self.W = uniform_initializer_variable(init_width, shape=[self.size, self.dim], name=name)
        else:
            self.size = w.shape[0]
            self.dim = w.shape[1]
            self.W = matrix_initialization_variable(w, name)

        self.l1_norm = tf.reduce_sum(tf.abs(self.W))
        self.l2_norm = tf.nn.l2_loss(self.W)

    def get_dim(self):
        return self.dim

    def get_value(self, session):
        return session.run([self.W])


class WordEmbedding(Embedding):
    def __init__(self, word2id, dimension=50, init_width=1.0, name="word_emb", verbose=True):
        self.vocab_size = len(word2id)
        super(WordEmbedding, self).__init__(size=self.vocab_size, dimension=dimension, init_width=init_width, name=name)

        self.word2id = word2id
        self.id2word = {idx: word for word, idx in word2id.items()}

        if verbose:
            logger.info("Word embedding finished initializing.\n")
            logger.info("Word size: %d\n" % self.vocab_size)
            logger.info("Word dimension: %d\n" % self.dim)

    def get_vocab_size(self):
        return self.vocab_size










