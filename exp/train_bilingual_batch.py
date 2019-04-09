import sys
import time
import os
import argparse
import io

import tensorflow as tf
import numpy as np
import logging

from src.config import BweConfig
from src.data_utils.prepare_data import read_data, build_dataset, save_vocab
from src.data_utils.prepare_data import read_parallel_sentence, sort_parallel_sentence, change_contexts_to_indexs
from src.data_utils.prepare_data import read_cross_word_weight, align_sen_batch, align_sentence_length
from src.data_utils.prepare_data import filter_cross_word_dict
from src.model import BilingualWordEmbedding
from src.utils import pre_logger, visualize, get_log_name, get_lines, visualize_training

__author__ = "Jocelyn"

FLAGS = None


def tune_bilingual_word_emb_hyperparameter(config_file, src_train_file, src_valid_file, tar_train_file, tar_valid_file,
                                           session):
    """
    :param config_file:
    :param src_train_file:
    :param src_valid_file:
    :param tar_train_file:
    :param tar_valid_file:
    :param session:
    :return:
    """
    # read config file and set logger
    bwe_config = BweConfig(config_file)
    logger_name = "de_en_alpha%.2f_beta%.2f_gama%.2f_lambda%.2f_epoch%d" % (bwe_config.alpha, bwe_config.beta,
                                                                            bwe_config.gama, bwe_config.lamb,
                                                                            bwe_config.epochs_pre_train)

    # logger = logging.getLogger(__name__)
    logger = pre_logger("train_embedding_" + logger_name)

    # read source file
    logger.info("Starting reading monolingual source file!\n")
    src_data = read_data(src_train_file)
    src_word2id, src_id2count, src_id2word = build_dataset(src_data, bwe_config.min_count)
    save_vocab(src_word2id, src_id2count, FLAGS.save_src_vocab)
    src_data = change_contexts_to_indexs(src_data, src_word2id)
    logger.info("Reading monolingual source file done: source vocab words=%d\n" % len(src_word2id))

    # read target file
    logger.info("Starting reading target monolingual file!\n")
    tar_data = read_data(tar_train_file)
    tar_word2id, tar_id2count, tar_id2word = build_dataset(tar_data, bwe_config.min_count)
    save_vocab(tar_word2id, tar_id2count, FLAGS.save_tar_vocab)
    tar_data = change_contexts_to_indexs(tar_data, tar_word2id)
    logger.info("Reading target monolingual file done: target vocab words=%d\n" % len(tar_word2id))

    # read valid source file
    logger.info("Start reading monolingual valid file!\n")
    src_valid_data = read_data(src_valid_file)
    src_valid_data = change_contexts_to_indexs(src_valid_data, src_word2id)

    tar_valid_data = read_data(tar_valid_file)
    tar_valid_data = change_contexts_to_indexs(tar_valid_data, tar_word2id)
    logger.info("Finish reading monolingual valid data!\n")

    # read parallel sentence
    logger.info("Starting reading parallel sentence file to buckets!\n")
    train_sentence = read_parallel_sentence(src_train_file, tar_train_file, src_word2id, tar_word2id)
    train_sentence = align_sen_batch(train_sentence, bwe_config.batch_size)
    src_train_sen = [sen[0] for sen in train_sentence]
    src_train_sen = align_sentence_length(src_train_sen, bwe_config.max_len, src_word2id)
    tar_train_sen = [sen[1] for sen in train_sentence]
    tar_train_sen = align_sentence_length(tar_train_sen, bwe_config.max_len, tar_word2id)
    logger.info("Reading parallel sentence file done: src sentence=%d, tar sentence=%d\n"
                % (len(src_train_sen), len(tar_train_sen)))

    # read valid parallel data
    logger.info("Start reading valid parallel data!\n")
    valid_sentence = read_parallel_sentence(src_valid_file, tar_valid_file, src_word2id, tar_word2id)
    valid_sentence = align_sen_batch(valid_sentence, bwe_config.batch_size)
    src_valid_sen = [sen[0] for sen in valid_sentence]
    src_valid_sen = align_sentence_length(src_valid_sen, bwe_config.max_len, src_word2id)
    tar_valid_sen = [sen[1] for sen in valid_sentence]
    tar_valid_sen = align_sentence_length(tar_valid_sen, bwe_config.max_len, tar_word2id)
    logger.info("Finish reading valid parallel sentence!\n")

    # read cross word weight
    logger.info("Start reading cross word weight\n")
    src2tar_word_weight = read_cross_word_weight(FLAGS.s2t_word_weight_file, src_word2id, tar_word2id)
    tar2src_word_weight = read_cross_word_weight(FLAGS.t2s_word_weight_file, tar_word2id, src_word2id)

    # filter cross word weight
    src2tar_word_weight = filter_cross_word_dict(src2tar_word_weight, 10)
    tar2src_word_weight = filter_cross_word_dict(tar2src_word_weight, 10)
    logger.info("Finish reading cross word weight\n")

    # train data
    batches_mono_words = int(max(len(src_data), len(tar_data)) / bwe_config.batch_size)
    # train_gap batches words training for one batch sentence training
    batches_bi = int(len(src_train_sen) / bwe_config.batch_size)
    batches_mono_bi = int(batches_mono_words / batches_bi)
    train_gap = batches_mono_bi
    logger.info("mono words training batch:%d\nbilingual training batch:%d\n"
                "batches of mono to bi:%d\ntraining gap:%d\n"
                % (batches_mono_words, batches_bi, batches_mono_bi, train_gap))

    src_words_to_train = (batches_mono_bi + 1) * batches_bi * bwe_config.epochs_to_train
    tar_words_to_train = (batches_mono_bi + 1) * batches_bi * bwe_config.epochs_to_train
    all_sen_to_train = batches_bi * bwe_config.epochs_to_train

    model = BilingualWordEmbedding(bwe_config, session, src_word2id, src_id2count, src_id2word, src_words_to_train,
                                   tar_word2id, tar_id2count, tar_id2word, tar_words_to_train, all_sen_to_train,
                                   name="bilingual_word_embedding")

    valid_size = 16
    valid_window = 100
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    summary_op = tf.summary.merge_all()
    # summary_writer = tf.summary.FileWriter(FLAGS.summary_path, model.session.graph)
    last_summary_time, last_checkpoint_time = 0, 0

    logger.info("Start training!\n")
    train_loss = []
    valid_loss = []
    for i in range(bwe_config.epochs_to_train):
        logger.info("Training epoch=%d\n" % i)

        loss_mono, loss_bi, loss_without_bi, total_loss = [], [], [], []
        for j in range(batches_bi):
            # train train_gap batches mono words / train one batch bilingual sentence
            # train monolingual
            for k in range(train_gap - 1):
                this_src_examples, this_src_labels, this_tar_examples, this_tar_labels = model.get_batch_mono(
                    src_data,
                    tar_data)
                loss = model.train_mono_step(this_src_examples, this_src_labels,
                                             this_tar_examples, this_tar_labels)
                this_loss_mono, this_loss_without_bi = loss[0], loss[1]

                now = time.time()
                loss_mono.append(this_loss_mono)
                loss_without_bi.append(this_loss_without_bi)

                if now - last_checkpoint_time >= bwe_config.checkpoint_interval:
                    step = i * batches_mono_words + j * train_gap + k
                    model.saver.save(model.session,
                                     os.path.join(FLAGS.save_checkpoint_path, "model.ckpt"),
                                     global_step=step)
                    last_checkpoint_time = now

            # train monolingual and bilingual
            # logger.info("Now train bilingual step=%d\n" % j)
            (this_src_examples, this_src_labels, this_tar_examples, this_tar_labels, this_src_sen, this_tar_sen,
             this_s2t_weight, this_t2s_weight) = model.get_batch(j, src_data, tar_data, src_train_sen, tar_train_sen,
                                                                 src2tar_word_weight, tar2src_word_weight)
            result = model.train_step(this_src_examples, this_src_labels, this_tar_examples, this_tar_labels,
                                      this_src_sen, this_tar_sen, this_s2t_weight, this_t2s_weight, summary_op, i)
            this_loss_mono, this_loss_bi, this_loss, summary_str = result[0], result[1], result[2], result[3]
            now = time.time()
            logger.info("epoch=%d, train step=%d, monolingual loss=%f, bilingual loss=%f, total loss=%f\n"
                        % (i, j, this_loss_mono, this_loss_bi, this_loss))
            sys.stdout.flush()

            # logger.info("attention score shape:%d, %d" % (len(att_scores), len(att_scores[0])))
            # logger.info("attention scores:%s\n" % str(att_scores))
            loss_mono.append(this_loss_mono)
            loss_bi.append(this_loss_bi)
            total_loss.append(this_loss)

            """
            if now - last_summary_time >= bwe_config.summary_interval:
                # summary_str = model.session.run(summary_op)
                summary_writer.add_summary(summary_str, i * batches_mono_words + j * train_gap + k)
                last_summary_time = now         
            """

            if now - last_checkpoint_time >= bwe_config.checkpoint_interval:
                step = i * batches_mono_words + (j + 1) * train_gap - 1
                model.saver.save(model.session,
                                 os.path.join(FLAGS.save_checkpoint_path, "model.ckpt"),
                                 global_step=step)
                last_checkpoint_time = now

        ave_loss_mono = np.average(loss_mono)
        ave_loss_bi = np.average(loss_bi)
        ave_loss_total = np.average(total_loss)
        logger.info("epoch=%d, average mono loss=%f, average bilingual loss=%f, average loss=%f\n"
                    % (i, ave_loss_mono, ave_loss_bi, ave_loss_total))

        if len(train_loss) > 2 and ave_loss_total > max(train_loss[-3:]):
            session.run(model.learning_rate_decay_op)
        train_loss.append(ave_loss_total)

        # validation data
        batches_valid_mono_words = int(max(len(src_valid_data), len(tar_valid_data)) / bwe_config.batch_size)
        batches_valid_bi = int(len(src_valid_sen) / bwe_config.batch_size)
        batches_valid_mono_bi = int(batches_valid_mono_words / batches_valid_bi)
        train_valid_gap = batches_valid_mono_bi

        epoch_valid_loss = []
        for j in range(batches_valid_bi):
            for k in range(train_valid_gap - 1):
                this_valid_src_examples, this_valid_src_labels, this_valid_tar_examples, this_valid_tar_labels = \
                    model.get_batch_mono(src_valid_data, tar_valid_data, True)
                loss = model.predict_mono_step(this_valid_src_examples, this_valid_src_labels, this_valid_tar_examples,
                                               this_valid_tar_labels)
                this_valid_loss_mono, this_valid_loss_without_bi = loss[0], loss[1]

            (this_valid_src_examples, this_valid_src_labels, this_valid_tar_examples, this_valid_tar_labels,
             this_valid_src_sen, this_valid_tar_sen, this_s2t_weight, this_t2s_weight) = \
                model.get_batch(j, src_valid_data, tar_valid_data, src_valid_sen, tar_valid_sen, src2tar_word_weight,
                                tar2src_word_weight, True)
            result = model.predict_step(this_valid_src_examples, this_valid_src_labels, this_valid_tar_examples,
                                        this_valid_tar_labels, this_valid_src_sen, this_valid_tar_sen, this_s2t_weight,
                                        this_t2s_weight, i)
            this_valid_loss_mono, this_valid_loss_bi, this_valid_loss = result[0], result[1], result[2]

            epoch_valid_loss.append(this_valid_loss)
        this_ave_valid_loss = np.average(epoch_valid_loss)
        logger.info("epoch=%d, valid loss=%f\n" % (i, this_ave_valid_loss))
        valid_loss.append(this_ave_valid_loss)

    logger.info("src word similarity:\n")
    test_similarity(model.src_word2vec.embedding.W, valid_dataset, valid_size, valid_examples, src_id2word, session,
                    logger)
    logger.info("Tar word similarity:\n")
    test_similarity(model.tar_word2vec.embedding.W, valid_dataset, valid_size, valid_examples, tar_id2word, session,
                    logger)
    logger.info("Src word cross lingual word similarity:\n")
    test_cross_lingual_similarity(model.src_word2vec.embedding.W, model.tar_word2vec.embedding.W, valid_dataset,
                                  valid_size, valid_examples, src_id2word, tar_id2word, session, logger)
    logger.info("Tar word cross lingual word similarity:\n")

    test_cross_lingual_similarity(model.tar_word2vec.embedding.W, model.src_word2vec.embedding.W, valid_dataset,
                                  valid_size, valid_examples, tar_id2word, src_id2word, session, logger)

    model.src_word2vec.save(FLAGS.save_src_embedding_path + "." + logger_name + ".vec")
    model.tar_word2vec.save(FLAGS.save_tar_embedding_path + "." + logger_name + ".vec")

    min_valid_loss = np.min(valid_loss)
    min_ind = int(np.argmin(valid_loss))
    min_train_loss = train_loss[min_ind]
    logger.info("Total Minimum:train loss=%f, valid loss=%f\n"
                % (min_train_loss, min_valid_loss))

    # visualize(train_loss, valid_loss)
    return [min_train_loss, min_valid_loss]


def train_bilingual_word_embedding(config_file, src_file, tar_file, session):
    """
    (train one batch word and update and train ,update until train one batch bilingual sentence) or train several
    batches words and one batch bilingual sentence and then update together
    :param config_file:
    :param src_file:
    :param tar_file:
    :param session:
    :return:
    """
    # read config file
    bwe_config = BweConfig(config_file)

    # set logger
    logger_name = "de_en_alpha%.2f_beta%.2f_gama%.2f_lambda%.2f_epoch%d" % (bwe_config.alpha, bwe_config.beta,
                                                                            bwe_config.gama, bwe_config.lamb,
                                                                            bwe_config.epochs_pre_train)

    # logger = logging.getLogger(__name__)
    logger = pre_logger("train_embedding_" + logger_name)

    # read source file
    logger.info("Starting reading monolingual source file!\n")
    src_data = read_data(src_file)
    src_word2id, src_id2count, src_id2word = build_dataset(src_data, bwe_config.min_count)
    save_vocab(src_word2id, src_id2count, FLAGS.save_src_vocab)
    src_data = change_contexts_to_indexs(src_data, src_word2id)
    logger.info("Reading monolingual source file done: source vocab words=%d\n" % len(src_word2id))

    # read target file
    logger.info("Starting reading target monolingual file!\n")
    tar_data = read_data(tar_file)
    tar_word2id, tar_id2count, tar_id2word = build_dataset(tar_data, bwe_config.min_count)
    save_vocab(tar_word2id, tar_id2count, FLAGS.save_tar_vocab)
    tar_data = change_contexts_to_indexs(tar_data, tar_word2id)
    logger.info("Reading target monolingual file done: target vocab words=%d\n" % len(tar_word2id))

    # read parallel sentence
    logger.info("Starting reading parallel sentence file to buckets!\n")
    train_sentence = read_parallel_sentence(src_file, tar_file, src_word2id, tar_word2id)
    train_sentence = align_sen_batch(train_sentence, bwe_config.batch_size)
    src_train_sen = [sen[0] for sen in train_sentence]
    src_train_sen = align_sentence_length(src_train_sen, bwe_config.max_len, src_word2id)
    tar_train_sen = [sen[1] for sen in train_sentence]
    tar_train_sen = align_sentence_length(tar_train_sen, bwe_config.max_len, tar_word2id)
    logger.info("Reading parallel sentence file done: src sentence=%d, tar sentence=%d\n"
                % (len(src_train_sen), len(tar_train_sen)))

    # read cross word weight
    logger.info("Start reading cross word weight\n")
    src2tar_word_weight = read_cross_word_weight(FLAGS.s2t_word_weight_file, src_word2id, tar_word2id)
    tar2src_word_weight = read_cross_word_weight(FLAGS.t2s_word_weight_file, tar_word2id, src_word2id)

    # filter cross word weight
    src2tar_word_weight = filter_cross_word_dict(src2tar_word_weight, 10)
    tar2src_word_weight = filter_cross_word_dict(tar2src_word_weight, 10)
    logger.info("Finish reading cross word weight\n")

    # train data
    batches_mono_words = int(max(len(src_data), len(tar_data)) / bwe_config.batch_size)
    # train_gap batches words training for one batch sentence training
    batches_bi = int(len(src_train_sen) / bwe_config.batch_size)
    batches_mono_bi = int(batches_mono_words / batches_bi)
    train_gap = batches_mono_bi
    logger.info("mono words training batch:%d\nbilingual training batch:%d\n"
                "batches of mono to bi:%d\ntraining gap:%d\n"
                % (batches_mono_words, batches_bi, batches_mono_bi, train_gap))

    src_words_to_train = (batches_mono_bi + 1) * batches_bi * bwe_config.epochs_to_train
    tar_words_to_train = (batches_mono_bi + 1) * batches_bi * bwe_config.epochs_to_train
    all_sen_to_train = batches_bi * bwe_config.epochs_to_train
    
    model = BilingualWordEmbedding(bwe_config, session, src_word2id, src_id2count, src_id2word, src_words_to_train,
                                   tar_word2id, tar_id2count, tar_id2word, tar_words_to_train, all_sen_to_train,
                                   name="bilingual_word_embedding")

    valid_size = 16
    valid_window = 100
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    summary_op = tf.summary.merge_all()
    # summary_writer = tf.summary.FileWriter(FLAGS.summary_path, model.session.graph)
    last_summary_time, last_checkpoint_time = 0, 0

    logger.info("Start training!\n")
    all_loss = []
    for i in range(bwe_config.epochs_to_train):
        logger.info("Training epoch=%d\n" % i)

        loss_mono, loss_bi, loss_without_bi, total_loss = [], [], [], []
        for j in range(batches_bi):
            # train train_gap batches mono words / train one batch bilingual sentence
            # train monolingual
            for k in range(train_gap - 1):
                this_src_examples, this_src_labels, this_tar_examples, this_tar_labels = model.get_batch_mono(
                    src_data,
                    tar_data)
                loss = model.train_mono_step(this_src_examples, this_src_labels,
                                             this_tar_examples, this_tar_labels)
                this_loss_mono, this_loss_without_bi = loss[0], loss[1]

                now = time.time()
                loss_mono.append(this_loss_mono)
                loss_without_bi.append(this_loss_without_bi)

                """
                if now - last_summary_time >= bwe_config.summary_interval:
                    # summary_str = model.session.run(summary_op)
                    summary_writer.add_summary(summary_str, i * batches_mono_words + j * train_gap + k)
                    last_summary_time = now         
                """

                if now - last_checkpoint_time >= bwe_config.checkpoint_interval:
                    step = i * batches_mono_words + j * train_gap + k
                    model.saver.save(model.session,
                                     os.path.join(FLAGS.save_checkpoint_path, "model.ckpt"),
                                     global_step=step)
                    last_checkpoint_time = now

            # train monolingual and bilingual
            # logger.info("Now train bilingual step=%d\n" % j)
            (this_src_examples, this_src_labels, this_tar_examples, this_tar_labels, this_src_sen, this_tar_sen,
             this_s2t_weight, this_t2s_weight) = model.get_batch(j, src_data, tar_data, src_train_sen, tar_train_sen,
                                                                 src2tar_word_weight, tar2src_word_weight)
            result = model.train_step(this_src_examples, this_src_labels, this_tar_examples, this_tar_labels,
                                      this_src_sen, this_tar_sen, this_s2t_weight, this_t2s_weight, summary_op, i)
            this_loss_mono, this_loss_bi, this_loss, summary_str = result[0], result[1], result[2], result[3]
            now = time.time()
            logger.info("epoch=%d, train step=%d, monolingual loss=%f, bilingual loss=%f, total loss=%f\n"
                        % (i, j, this_loss_mono, this_loss_bi, this_loss))
            sys.stdout.flush()

            # logger.info("attention score shape:%d, %d" % (len(att_scores), len(att_scores[0])))
            # logger.info("attention scores:%s\n" % str(att_scores))
            loss_mono.append(this_loss_mono)
            loss_bi.append(this_loss_bi)
            total_loss.append(this_loss)

            """
            if now - last_summary_time >= bwe_config.summary_interval:
                # summary_str = model.session.run(summary_op)
                summary_writer.add_summary(summary_str, i * batches_mono_words + j * train_gap + k)
                last_summary_time = now         
            """

            if now - last_checkpoint_time >= bwe_config.checkpoint_interval:
                step = i * batches_mono_words + (j + 1) * train_gap - 1
                model.saver.save(model.session,
                                 os.path.join(FLAGS.save_checkpoint_path, "model.ckpt"),
                                 global_step=step)
                last_checkpoint_time = now

        ave_loss_mono = np.average(loss_mono)
        ave_loss_bi = np.average(loss_bi)
        ave_loss_total = np.average(total_loss)
        logger.info("epoch=%d, average mono loss=%f, average bilingual loss=%f, average loss=%f\n"
                    % (i, ave_loss_mono, ave_loss_bi, ave_loss_total))

        if len(all_loss) > 2 and ave_loss_total > max(all_loss[-3:]):
            session.run(model.learning_rate_decay_op)
        all_loss.append(ave_loss_total)

    """
    logger.info("src word similarity:\n")
    test_similarity(model.src_word2vec.embedding.W, valid_dataset, valid_size, valid_examples, src_id2word,
                    session, logger)
    logger.info("Tar word similarity:\n")
    test_similarity(model.tar_word2vec.embedding.W, valid_dataset, valid_size, valid_examples, tar_id2word,
                    session, logger)
    logger.info("Src word cross lingual word similarity:\n")
    test_cross_lingual_similarity(model.src_word2vec.embedding.W, model.tar_word2vec.embedding.W, valid_dataset,
                                  valid_size, valid_examples, src_id2word, tar_id2word, session, logger)
    logger.info("Tar word cross lingual word similarity:\n")

    test_cross_lingual_similarity(model.tar_word2vec.embedding.W, model.src_word2vec.embedding.W, valid_dataset,
                                  valid_size, valid_examples, tar_id2word, src_id2word, session, logger)
    """

    model.src_word2vec.save(FLAGS.save_src_embedding_path + "." + logger_name + ".vec")
    model.tar_word2vec.save(FLAGS.save_tar_embedding_path + "." + logger_name + ".vec")

    # visualize training loss
    # visualize_training(all_loss)


def test_similarity(embeddings, valid_dataset, valid_size, valid_examples, id2word, session, logger):
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm

    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
    sim = session.run(similarity)
    for i in range(valid_size):
        valid_word = id2word[valid_examples[i]]
        top_k = 8
        nearest = (-sim[i, :]).argsort()[1: top_k + 1]
        log_str = "Nearest to %s:" % valid_word
        for k in range(top_k):
            close_word = id2word[nearest[k]]
            log_str = "%s %s," % (log_str, close_word)
        logger.info("%s\n" % log_str)


def test_cross_lingual_similarity(src_embeddings, tar_embeddings, valid_dataset, valid_size, valid_examples,
                                  src_id2word, tar_id2word, session, logger):
    norm_src = tf.sqrt(tf.reduce_sum(tf.square(src_embeddings), 1, keep_dims=True))
    normalized_src_embeddings = src_embeddings / norm_src
    norm_tar = tf.sqrt(tf.reduce_sum(tf.square(tar_embeddings), 1, keep_dims=True))
    normalized_tar_emebddings = tar_embeddings / norm_tar

    valid_embeddings = tf.nn.embedding_lookup(normalized_src_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_tar_emebddings, transpose_b=True)
    sim = session.run(similarity)
    for i in range(valid_size):
        valid_word = src_id2word[valid_examples[i]]
        top_k = 8
        nearest = (-sim[i, :]).argsort()[: top_k]
        log_str = "Nearest to %s: " % valid_word
        for k in range(top_k):
            close_word = tar_id2word[nearest[k]]
            log_str = "%s %s, " % (log_str, close_word)
        logger.info("%s\n" % log_str)


def grid_search_tune():
    """
    grad search to tune hyper parameter
    :return:
    """
    config_name = FLAGS.config_file
    f = open(config_name, "r", encoding="utf-8")
    lines = f.readlines()
    alpha = 0.05
    beta = 1 - alpha
    while alpha < 1.0:
        f = open(config_name, "w", encoding="utf-8")
        for line in lines:
            if "alpha=" in line:
                line = "alpha=" + str(alpha) + "\n"
            if "beta=" in line:
                line = "beta=" + str(beta) + "\n"
            f.write(line)
        f.close()

        g = tf.Graph()
        with g.as_default():
            sess = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=True))
            tune_bilingual_word_emb_hyperparameter(FLAGS.config_file, FLAGS.src_file, FLAGS.src_valid_file,
                                                   FLAGS.tar_file, FLAGS.tar_valid_file, sess)
            sess.close()
        g.finalize()

        alpha += 0.05
        beta = 1.0 - alpha


def tune_epochs_pre_train():
    config_name = FLAGS.config_file
    f = open(config_name, "r", encoding="utf-8")
    lines = f.readlines()
    for i in range(4, 26):
        f = open(config_name, "w", encoding="utf-8")
        for line in lines:
            if "epochs_pre_train=" in line:
                line = "epochs_pre_train=" + str(i) + "\n"
            f.write(line)
        f.close()

        g = tf.Graph()
        with g.as_default():
            sess = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=True))
            train_bilingual_word_embedding(FLAGS.config_file, FLAGS.src_file, FLAGS.tar_file, sess)
            sess.close()
        g.finalize()


def main(_):
    with tf.device("/gpu:0"):
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True))
        train_bilingual_word_embedding(FLAGS.config_file, FLAGS.src_file, FLAGS.tar_file, sess)
        """
        tune_bilingual_word_emb_hyperparameter(FLAGS.config_file, FLAGS.src_file, FLAGS.src_valid_file,
                                               FLAGS.tar_file, FLAGS.tar_valid_file, sess)
        """


if __name__ == "__main__":
    path = os.path.dirname(os.path.dirname(os.path.abspath("train_bilingual_batch.py")))
    exp_path = os.path.join(os.path.dirname(os.path.dirname(path)), "emnlp\\experiment")
    parse = argparse.ArgumentParser()

    parse.add_argument("--save_src_vocab", type=str,
                       default=os.path.join(exp_path, "AER\\bibles_eu_vocab\\fr.vocab"),
                       help="file path for saving source vocab")
    parse.add_argument("--save_tar_vocab", type=str,
                       default=os.path.join(exp_path, "AER\\bibles_eu_vocab\\en.vocab"),
                       help="file path for saving target vocab")
    parse.add_argument("--summary_path", type=str,
                       default=os.path.join(path, "data\\de-en\\summary"),
                       help="file path for saving training summary")
    parse.add_argument("--save_checkpoint_path", type=str,
                       default=os.path.join(path, "data\\de-en\\model"),
                       help="directory for saving checkpoint")
    parse.add_argument("--save_src_embedding_path", type=str,
                       default=os.path.join(exp_path, "AER\\bibles_eu_emb\\en-fr\\fr.bwe8.emb"),
                       help="file path for saving source embedding")
    parse.add_argument("--save_tar_embedding_path", type=str,
                       default=os.path.join(exp_path, "AER\\bibles_eu_emb\\en-fr\\en.bwe8.emb"),
                       help="file path for saving target embedding")
    parse.add_argument("--config_file", type=str,
                       default=os.path.join(path, "conf\\bwe8.conf"),
                       help="file path for configuration")
    parse.add_argument("--src_file", type=str,
                       default=os.path.join(exp_path, "AER\\union_data\\fr0.txt"),
                       help="file path for source file")
    parse.add_argument("--tar_file", type=str,
                       default=os.path.join(exp_path, "AER\\union_data\\en0.txt"),
                       help="file path for target file")
    parse.add_argument("--src_valid_file", type=str,
                       default=os.path.join(path, "data\\de-en\\tune\\valid.de.raw"),
                       help="file path for source valid file")
    parse.add_argument("--tar_valid_file", type=str,
                       default=os.path.join(path, "data\\de-en\\tune\\valid.en.raw"),
                       help="file path for target valid file")
    parse.add_argument("--s2t_word_weight_file", type=str,
                       default=os.path.join(exp_path, "AER\\union_word_idf_count\\fr2en.word.idf.count"),
                       help="File for storing cross word weight which is adding idf weight score")
    parse.add_argument("--t2s_word_weight_file", type=str,
                       default=os.path.join(exp_path, "AER\\union_word_idf_count\\en2fr.word.idf.count"))

    FLAGS, unparsed = parse.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]]+unparsed)


















