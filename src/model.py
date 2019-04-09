import tensorflow as tf
import random
import numpy as np

from src.word2vec import Word2Vec
from src.utils import get_sen_mask, truncated_normal_initializer_variable, zero_initializer_variable
from src.utils import sample_indexes

__author__ = "Jocelyn"


class BilingualWordEmbedding(object):
    def __init__(self, bwe_config, session, src_word2id, src_id2count, src_id2word, src_words_to_train,
                 tar_word2id, tar_id2count, tar_id2word, tar_words_to_train, all_train_sen, name=""):
        self.laplace = 0.00001
        self.config = bwe_config
        self.session = session
        self.max_len = self.config.max_len
        self.train_sentence = 0
        self.all_train_sentence = all_train_sen
        self.name = name

        self.src_word2vec = Word2Vec(bwe_config, session, src_word2id, src_id2count, src_id2word, src_words_to_train,
                                     "src_word2vec")
        self.tar_word2vec = Word2Vec(bwe_config, session, tar_word2id, tar_id2count, tar_id2word, tar_words_to_train,
                                     "tar_word2vec")

        # [batch, max_len]
        self.src_sen = tf.placeholder(dtype=tf.int32, shape=[self.config.batch_size, self.config.max_len],
                                      name=name+"_src_sen")
        # [batch, max_len]
        self.tar_sen = tf.placeholder(dtype=tf.int32, shape=[self.config.batch_size, self.config.max_len],
                                      name=name+"_tar_sen")

        self.s2t_weight = tf.placeholder(dtype=tf.float32,
                                         shape=[self.config.batch_size, self.config.max_len, self.config.max_len],
                                         name=name+"_s2t_weight")
        self.t2s_weight = tf.placeholder(dtype=tf.float32,
                                         shape=[self.config.batch_size, self.config.max_len, self.config.max_len],
                                         name=name+"_t2s_weight")

        self.loss_mono = self.src_word2vec.loss_nce + self.tar_word2vec.loss_nce
        self.loss_without_bi = self.config.alpha * self.loss_mono
        self.loss_bi_1 = self.cross_lingual_similarity_1()
        self.loss_1 = self.config.alpha * self.loss_mono + self.config.beta * self.loss_bi_1

        self.loss_bi_2, self.loss_coverage, self.loss_sparsity = self.cross_lingual_similarity_2()
        # self.loss_2 = self.config.alpha * self.loss_mono + self.config.beta * self.loss_bi_2
        self.loss_2 = self.config.alpha * self.loss_mono + self.config.beta * self.loss_bi_2 \
                    + self.config.gama * self.loss_coverage + self.config.lamb * self.loss_sparsity

        tf.summary.scalar("loss_mono", self.loss_mono)
        tf.summary.scalar("loss_bi_1", self.loss_bi_1)
        tf.summary.scalar("loss_coverage", self.loss_coverage)
        tf.summary.scalar("loss_sparsity", self.loss_sparsity)
        tf.summary.scalar("loss_without_bi", self.loss_without_bi)
        tf.summary.scalar("total_loss_1", self.loss_1)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.lr = tf.Variable(self.config.optimizer_config.lr, trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.lr.assign(self.lr * self.config.learning_rate_decay_factor)
        self.train_1 = self.optimize(self.loss_1)
        self.train_2 = self.optimize(self.loss_2)
        self.train_mono = self.optimize(self.loss_without_bi)
        # self.summary_op = tf.summary.merge_all()

        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def normalize_weight_score(self, src_align_score):
        align_score_total = tf.reduce_sum(src_align_score, axis=-1, keep_dims=True) + self.laplace
        src_align_score = src_align_score / align_score_total
        return src_align_score

    def compute_align_score(self, src_embs, tar_embs):
        src_norm = tf.sqrt(tf.reduce_sum(tf.square(src_embs), axis=-1, keep_dims=True))
        src_embs = src_embs / src_norm
        tar_norm = tf.sqrt(tf.reduce_sum(tf.square(tar_embs), axis=-1, keep_dims=True))
        tar_embs = tar_embs / tar_norm

        src_embs_split = tf.split(src_embs, num_or_size_splits=self.config.batch_size, axis=0)
        src_embs_split = [tf.reshape(split, [self.config.max_len, self.config.embedding_size])
                          for split in src_embs_split]
        tar_embs_split = tf.split(tar_embs, num_or_size_splits=self.config.batch_size, axis=0)
        tar_embs_split = [tf.reshape(split, [self.config.max_len, self.config.embedding_size])
                          for split in tar_embs_split]

        # [batch, max_len, max_len]
        sim_scores = [tf.matmul(src_split, tar_split, transpose_b=True)
                      for src_split, tar_split in zip(src_embs_split, tar_embs_split)]
        sim_scores = [tf.nn.relu(scores) for scores in sim_scores]
        sim_scores = [self.normalize_weight_score(scores) for scores in sim_scores]
        # sim_scores = tf.reshape(sim_scores, [self.config.batch_size, self.config.max_len, self.config.max_len])
        return sim_scores

    def compute_coverage_loss(self, align_score):
        """
        compute coverage loss according to each word
        :param align_score: batch size: [max_len, max_len]
        :return:
        """
        # [batch, ]
        coverage_loss_split = [tf.reduce_sum(tf.square(1.0 - tf.reduce_sum(score, axis=0))) for score in align_score]
        coverage_loss_split = [score / self.config.max_len for score in coverage_loss_split]
        coverage_loss = tf.reduce_sum(coverage_loss_split)
        return coverage_loss

    def compute_sparsity_loss(self, align_score):
        """
        compute sparsity loss according to each word
        :param align_score:  batch size: [max_len, max_len]
        :return:
        """
        sparsity_loss_split = [tf.reduce_sum(tf.abs(split)) for split in align_score]
        sparsity_loss_split = [score / self.config.max_len for score in sparsity_loss_split]
        sparsity_loss = tf.reduce_sum(sparsity_loss_split)
        return sparsity_loss

    def compute_sim_loss_1(self, src_embedding, tar_embedding, src_sen, tar_sen, s2t_weight):
        """
        compute similarity score
        :param src_embedding:
        :param tar_embedding:
        :param src_sen: [batch, max_len]
        :param tar_sen: [batch, max_len]
        :param s2t_weight:[batch, max_len, max_len]
        :return:
        """
        src_embs = tf.nn.embedding_lookup(src_embedding, src_sen)  # [batch, max_len, dimension]
        tar_embs = tf.nn.embedding_lookup(tar_embedding, tar_sen)  # [batch, max_len, dimension]

        align_score = tf.split(s2t_weight, num_or_size_splits=self.config.batch_size, axis=0)
        align_score = [tf.reshape(split, [self.config.max_len, self.config.max_len]) for split in align_score]
        align_score = [self.normalize_weight_score(split) for split in align_score]

        # batch size:[max_len, embedding_size]
        tar_embedding_split = tf.split(tar_embs, num_or_size_splits=self.config.batch_size, axis=0)
        tar_embedding_split = [tf.reshape(split, [self.config.max_len, self.config.embedding_size])
                               for split in tar_embedding_split]

        # batch size:[max_len, embedding_size]
        tar_emb_map = [tf.matmul(score, split) for score, split in zip(align_score, tar_embedding_split)]
        tar_emb_map = tf.reshape(tar_emb_map, [self.config.batch_size, self.config.max_len, self.config.embedding_size])
        sim_score = tf.reduce_sum(tf.square(src_embs - tar_emb_map), axis=-1)  # [batch, max_len]
        sim_loss = tf.reduce_sum(sim_score) / self.config.max_len
        return sim_loss

    def cross_lingual_similarity_1(self):
        """
        compute the bilingual similarity loss
        :return:
        """
        src_loss = self.compute_sim_loss_1(self.src_word2vec.embedding.W, self.tar_word2vec.embedding.W,
                                           self.src_sen, self.tar_sen, self.s2t_weight)
        tar_loss = self.compute_sim_loss_1(self.tar_word2vec.embedding.W, self.src_word2vec.embedding.W,
                                           self.tar_sen, self.src_sen, self.t2s_weight)

        bi_loss = (src_loss + tar_loss) / self.config.batch_size
        return bi_loss

    def compute_sim_loss_2(self, src_embedding, tar_embedding, src_sen, tar_sen):
        """
        compute similarity score
        :param src_embedding:
        :param tar_embedding:
        :param src_sen: [batch, max_len]
        :param tar_sen: [batch, max_len]
        :return:
        """
        src_embs = tf.nn.embedding_lookup(src_embedding, src_sen)  # [batch, max_len, dimension]
        tar_embs = tf.nn.embedding_lookup(tar_embedding, tar_sen)  # [batch, max_len, dimension]

        align_score = self.compute_align_score(src_embs, tar_embs)  # batch size: [max_len, max_len]

        # batch size:[max_len, embedding_size]
        tar_embedding_split = tf.split(tar_embs, num_or_size_splits=self.config.batch_size, axis=0)
        tar_embedding_split = [tf.reshape(split, [self.config.max_len, self.config.embedding_size])
                               for split in tar_embedding_split]

        # batch size:[max_len, embedding_size]
        tar_emb_map = [tf.matmul(score, split) for score, split in zip(align_score, tar_embedding_split)]
        tar_emb_map = tf.reshape(tar_emb_map, [self.config.batch_size, self.config.max_len, self.config.embedding_size])
        sim_score = tf.reduce_sum(tf.square(src_embs - tar_emb_map), axis=-1)  # [batch, max_len]
        sim_loss = tf.reduce_sum(sim_score) / self.config.max_len

        # coverage loss
        coverage_loss = self.compute_coverage_loss(align_score)
        # sparsity loss
        sparsity_loss = self.compute_sparsity_loss(align_score)
        return sim_loss, sparsity_loss, coverage_loss

    def cross_lingual_similarity_2(self):
        """
        compute the bilingual similarity loss
        :return:
        """
        src_loss, src_coverage_loss, src_sparsity_loss = self.compute_sim_loss_2(self.src_word2vec.embedding.W,
                                                                                 self.tar_word2vec.embedding.W,
                                                                                 self.src_sen, self.tar_sen)
        tar_loss, tar_coverage_loss, tar_sparsity_loss = self.compute_sim_loss_2(self.tar_word2vec.embedding.W,
                                                                                 self.src_word2vec.embedding.W,
                                                                                 self.tar_sen, self.src_sen)
        bi_loss = (src_loss + tar_loss) / self.config.batch_size
        coverage_loss = (src_coverage_loss + tar_coverage_loss) / self.config.batch_size
        sparsity_loss = (src_sparsity_loss + tar_sparsity_loss) / self.config.batch_size
        return bi_loss, coverage_loss, sparsity_loss

    def optimize(self, loss):
        """
        optimizer = self.config.optimizer_config.get_optimizer()
        train = optimizer.minimize(loss)
        lr = self.config.optimizer_config.lr * tf.maximum(0.0001
                                                   1.0 - float(self.train_sentence) / self.all_train_sentence)
        self.lr = lr
        """
        optimizer = self.config.optimizer_config.get_optimizer(self.lr)
        train = optimizer.minimize(loss)
        return train

    @property
    def learning_rate(self):
        return self.lr

    def train_mono_step(self, this_src_examples, this_src_labels, this_tar_examples, this_tar_labels):
        input_feed = {self.src_word2vec.examples: this_src_examples,
                      self.src_word2vec.labels: this_src_labels,
                      self.tar_word2vec.examples: this_tar_examples,
                      self.tar_word2vec.labels: this_tar_labels}
        output_feed = [self.train_mono, self.loss_mono, self.loss_without_bi]
        _, loss_mono, loss_without_bi = self.session.run(output_feed, input_feed)
        return [loss_mono, loss_without_bi]

    def predict_mono_step(self, this_src_examples, this_src_labels, this_tar_examples, this_tar_labels):
        input_feed = {self.src_word2vec.examples: this_src_examples,
                      self.src_word2vec.labels: this_src_labels,
                      self.tar_word2vec.examples: this_tar_examples,
                      self.tar_word2vec.labels: this_tar_labels}
        output_feed = [self.loss_mono, self.loss_without_bi]
        loss_mono, loss_without_bi = self.session.run(output_feed, input_feed)
        return [loss_mono, loss_without_bi]

    def train_step(self, this_src_examples, this_src_labels, this_tar_examples, this_tar_labels,
                   this_src_sen, this_tar_sen, this_s2t_weight, this_t2s_weight, summary_op, epoch):
        input_feed = {self.src_word2vec.examples: this_src_examples,
                      self.src_word2vec.labels: this_src_labels,
                      self.tar_word2vec.examples: this_tar_examples,
                      self.tar_word2vec.labels: this_tar_labels,
                      self.src_sen: this_src_sen,
                      self.tar_sen: this_tar_sen,
                      self.s2t_weight: this_s2t_weight,
                      self.t2s_weight: this_t2s_weight}

        if epoch < self.config.epochs_pre_train:
            output_feed = [self.train_1, self.loss_mono, self.loss_bi_1, self.loss_1, summary_op]
        else:
            output_feed = [self.train_2, self.loss_mono, self.loss_bi_2, self.loss_2, summary_op]
        # output_feed = [self.train, self.loss_mono, self.loss_bi,  self.loss, summary_op]
        _, loss_mono, loss_bi, loss, summary_str = self.session.run(output_feed, input_feed)
        return [loss_mono, loss_bi, loss, summary_str]

    def predict_step(self, this_src_examples, this_src_labels, this_tar_examples, this_tar_labels,
                     this_src_sen, this_tar_sen, this_s2t_weight, this_t2s_weight, epoch):
        input_feed = {self.src_word2vec.examples: this_src_examples,
                      self.src_word2vec.labels: this_src_labels,
                      self.tar_word2vec.examples: this_tar_examples,
                      self.tar_word2vec.labels: this_tar_labels,
                      self.src_sen: this_src_sen,
                      self.tar_sen: this_tar_sen,
                      self.s2t_weight: this_s2t_weight,
                      self.t2s_weight: this_t2s_weight}

        if epoch < self.config.epochs_pre_train:
            output_feed = [self.loss_mono, self.loss_bi_1, self.loss_1]
        else:
            output_feed = [self.loss_mono, self.loss_bi_2, self.loss_2]
        # output_feed = [self.loss_mono, self.loss_bi, self.loss]
        loss_mono, loss_bi, loss = self.session.run(output_feed, input_feed)
        return [loss_mono, loss_bi, loss]

    def get_batch_mono(self, src_data, tar_data, valid=False):
        this_src_examples, this_src_labels = self.src_word2vec.get_batch(src_data, valid)
        this_tar_examples, this_tar_labels = self.tar_word2vec.get_batch(tar_data, valid)
        return (this_src_examples, this_src_labels,
                this_tar_examples, this_tar_labels)

    def get_line_weight(self, src, tar, s2t_weight_dict):
        line_weight = []
        for src_word in src:
            if src_word not in s2t_weight_dict.keys():
                word_weight = np.zeros(shape=[self.config.max_len], dtype=float)
            else:
                word_dict = s2t_weight_dict[src_word]
                word_weight = []
                for tar_word in tar:
                    if tar_word in word_dict.keys():
                        word_weight.append(word_dict[tar_word])
                    else:
                        word_weight.append(0.0)
            line_weight.append(word_weight)
        return line_weight

    def get_weight(self, s2t_weight_dict, src_sen, tar_sen):
        this_weight = []
        for src, tar in zip(src_sen, tar_sen):
            line_weight = self.get_line_weight(src, tar, s2t_weight_dict)
            this_weight.append(line_weight)
        return this_weight

    def get_batch(self, ind_bi, src_data, tar_data, src_train_sen, tar_train_sen, src2tar_weight_dict,
                  tar2src_weight_dict, valid=False):
        self.train_sentence += 1
        this_src_examples, this_src_labels = self.src_word2vec.get_batch(src_data, valid)
        this_tar_examples, this_tar_labels = self.tar_word2vec.get_batch(tar_data, valid)

        this_src_sen = src_train_sen[ind_bi * self.config.batch_size: (ind_bi + 1) * self.config.batch_size]
        this_tar_sen = tar_train_sen[ind_bi * self.config.batch_size: (ind_bi + 1) * self.config.batch_size]

        this_s2t_weight = self.get_weight(src2tar_weight_dict, this_src_sen, this_tar_sen)
        this_t2s_weight = self.get_weight(tar2src_weight_dict, this_tar_sen, this_src_sen)
        return (this_src_examples, this_src_labels,
                this_tar_examples, this_tar_labels,
                this_src_sen, this_tar_sen,
                this_s2t_weight, this_t2s_weight)




















