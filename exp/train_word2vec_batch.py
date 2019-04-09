import argparse
import sys
import time
import os

import tensorflow as tf
import numpy as np

from src.data_utils.prepare_data import read_data, build_dataset, save_vocab, word_skip_gram, align_batch_size
from src.config import BweConfig
from src.word2vec import Word2Vec

__author__ = "Jocelyn"

FLAGS = None


def word2vec_tune_hyperparameter(src_file, valid_file, config_file):
    bwe_config = BweConfig(config_file)

    src_data = read_data(src_file)
    word2id, id2count, id2word = build_dataset(src_data, bwe_config.min_count)
    # train file
    src_examples, src_labels = word_skip_gram(src_file, len(src_data), word2id, id2count, bwe_config.window_size,
                                              bwe_config.sub_sample)
    src_examples, src_labels = align_batch_size(src_examples, src_labels, bwe_config.batch_size)

    # valid file
    valid_data = read_data(valid_file)
    valid_examples, valid_labels = word_skip_gram(valid_file, len(valid_data), word2id, bwe_config.window_size,
                                                  bwe_config.sub_sample)

    session = tf.Session()
    src_model = Word2Vec(bwe_config, session, word2id, id2count, id2word, "src_word2vec")

    batches = len(src_examples) / bwe_config.batch_size

    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.save_summary_path, src_model.session.graph)
    last_words, last_time, last_summary_time = src_model.train_words, time.time(), 0
    last_checkpoint_time = 0

    train_loss = []
    valid_loss = []
    for i in range(bwe_config.epochs_to_train):
        loss = []
        for j in range(batches):
            time.sleep(bwe_config.statistics_interval)
            this_examples, this_labels = src_model.get_batch(i, src_examples, src_labels)
            train_loss = src_model.train_step(this_examples, this_labels)

            now = time.time()
            last_words, last_time, rate = src_model.train_words, now, \
                                          (src_model.train_words - last_words) / (now - last_time)
            print("epoch %d, batch %d, the training loss is: %f, lr=%f, words/sec=%f\n"
                  % (i + 1, j + 1, train_loss, src_model.session.run(src_model.lr), rate))
            sys.stdout.flush()

            if now - last_summary_time > bwe_config.summary_interval:
                summary_str = src_model.session.run(summary_op)
                summary_writer.add_summary(summary_str, i * batches + j)
                last_summary_time = now

            if now - last_checkpoint_time > bwe_config.checkpoint_interval:
                step = i * batches + j
                src_model.saver.save(src_model.session,
                                     os.path.join(FLAGS.save_checkpoint_path, "model.ckpt"),
                                     global_step=step)
                last_checkpoint_time = now

            loss.append(train_loss)
        ave_loss = np.average(loss)
        print("epoch %d, the average loss is: %f\n" % ave_loss)
        train_loss.append(ave_loss)

        valid_batches = len(valid_examples) // bwe_config.batch_size
        this_valid_loss = []
        for j in range(valid_batches):
            this_valid_examples, this_valid_labels = src_model.get_batch(j, valid_examples, valid_labels)
            v_loss = src_model.predict_step(this_valid_examples, this_valid_labels)
            this_valid_loss.append(v_loss)
        this_ave_valid_loss = np.average(this_valid_loss)
        print("epoch=%d, valid loss=%f\n" % (i, this_ave_valid_loss))
        valid_loss.append(this_ave_valid_loss)

    min_valid_loss = np.min(valid_loss)
    min_ind = np.argmin(valid_loss)
    min_train_loss = train_loss[min_ind[0]]
    print("Total: train loss=%f, valid loss=%f" % (min_train_loss, min_valid_loss))
    return min_train_loss, min_valid_loss


def word2vec_train_batch(src_file, config_file, save_word_file):
    bwe_config = BweConfig(config_file)

    src_data = read_data(src_file)
    word2id, id2count, id2word = build_dataset(src_data, bwe_config.min_count)
    save_vocab(word2id, id2count, save_word_file)

    src_examples, src_labels = word_skip_gram(src_file, len(src_data), word2id, id2count, bwe_config.window_size,
                                              bwe_config.sub_sample)
    src_examples, src_labels = align_batch_size(src_examples, src_labels, bwe_config.batch_size)

    session = tf.Session()
    src_model = Word2Vec(bwe_config, session, word2id, id2count, id2word, "src_word2vec")

    batches = len(src_examples) / bwe_config.batch_size

    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.save_summary_path, src_model.session.graph)
    last_words, last_time, last_summary_time = src_model.train_words, time.time(), 0
    last_checkpoint_time = 0

    for i in range(bwe_config.epochs_to_train):
        loss = []
        for j in range(batches):
            time.sleep(bwe_config.statistics_interval)
            this_examples, this_labels = src_model.get_batch(i, src_examples, src_labels)
            train_loss = src_model.train_step(this_examples, this_labels)

            now = time.time()
            last_words, last_time, rate = src_model.train_words, now, \
                                          (src_model.train_words - last_words) / (now - last_time)
            print("epoch %d, batch %d, the training loss is: %f, lr=%f, words/sec=%f\n"
                  % (i+1, j+1, train_loss, src_model.session.run(src_model.lr), rate))
            sys.stdout.flush()

            if now - last_summary_time > bwe_config.summary_interval:
                summary_str = src_model.session.run(summary_op)
                summary_writer.add_summary(summary_str, i * batches + j)
                last_summary_time = now

            if now - last_checkpoint_time > bwe_config.checkpoint_interval:
                step = i * batches + j
                src_model.saver.save(src_model.session,
                                     os.path.join(FLAGS.save_checkpoint_path, "model.ckpt"),
                                     global_step=step)
                last_checkpoint_time = now

            loss.append(train_loss)
        ave_loss = np.average(loss)
        print("epoch %d, the average loss is: %f\n" % ave_loss)

    # save word embedding
    src_model.save(os.path.join(FLAGS.save_embedding_path, "src.embedding"))


def main(_):
    word2vec_train_batch(FLAGS.src_file, FLAGS.config_file, FLAGS.save_vocab)


if __name__ == "__main__":
    parse = argparse.ArgumentParser()

    parse.add_argument("--src_file", type=str, default="data\\src.file", help="directory for source file")
    parse.add_argument("--config_file", type=str, default="conf\\bwe.conf", help="config file path")
    parse.add_argument("--save_word_file", type=str, default="data\\save_src_vocab",
                       help="file name for saving source file")
    parse.add_argument("--save_embedding_path", type=str, default="data\\embedding",
                       help="directory for saving embedding")
    parse.add_argument("--save_summary_path", type=str, default="data\\summary", help="directory for saving summary")
    parse.add_argument("--save_checkpoint_path", type=str, default="data\\checkpoint",
                       help="directory for saving checkpoint")

    FLAGS, unparsed = parse.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)





