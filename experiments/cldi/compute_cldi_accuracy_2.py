import os
import argparse
import sys
import random

import numpy as np
import tensorflow as tf

from src.utils import read_stop_words_lower

__author__ = "Jocelyn"
FLAGS = None


def read_words_freq(filename):
    """
    filter words according to their frequency
    :param filename:
    :return:
    """
    words = []
    f = open(filename, "r", encoding="utf-8")
    for line in f.readlines():
        lemmas = line.strip().split("|||")
        word = lemmas[0].strip().lower()
        words.append(word)
    return words


def filter_stop_words(freq_words, filename):
    stop_words = read_stop_words_lower(filename)
    new_freq_words = []
    for word in freq_words:
        if word in stop_words:
            continue
        if word.isdigit():
            continue
        if len(word) <= 1:
            continue
        new_freq_words.append(word)
    return new_freq_words


def filter_gold_word_synset(filename, src_freq_words, tar_freq_words):
    """
    :param filename:
    :param src_freq_words:
    :param tar_freq_words:
    :return:
    """
    gold_pair = []
    f = open(filename, "r", encoding="utf-8")
    for line in f.readlines():
        words = line.strip().split("|||")
        src = words[0].strip().lower()
        tar = words[1].strip().lower()
        if src in src_freq_words and tar in tar_freq_words:
            gold_pair.append([src, tar])
    return gold_pair


def read_embedding(filename, freq_words):
    """
    only read words whose frequency are greater than 1000
    :param filename:
    :param freq_words:
    :return:
    """
    words2id = {}
    id2words = []
    id2embedding = []
    f = open(filename, "r", encoding="utf-8")
    for line in f.readlines():
        lemmas = line.strip().split()
        word = lemmas[0]
        if word not in freq_words:
            continue
        embedding = []
        for lemma in lemmas[1:]:
            embedding.append(float(lemma.strip()))
        words2id[word] = len(words2id)
        id2words.append(word)
        id2embedding.append(embedding)
    return words2id, id2words, id2embedding


def filter_freq_words_embedding(freqwords, words2id):
    words = words2id.keys()
    new_freq_words = []
    for word in freqwords:
        if word in words:
            new_freq_words.append(word)
    return new_freq_words


def compute_neighbors(src_words2id, src_id2embedding, tar_id2embedding, src_word, k, session):
    src_word_id = src_words2id[src_word]
    src_word = src_id2embedding[src_word_id]
    # similarity_scores = np.dot(src_word, np.transpose(tar_id2embedding))

    dot = np.dot(src_word, np.transpose(tar_id2embedding))
    square1 = np.sqrt(np.dot(src_word, np.transpose(src_word)))
    square2 = np.diag(np.sqrt(np.dot(tar_id2embedding, np.transpose(tar_id2embedding))))
    similarity_scores = dot / (square1 * square2)

    similarity_scores = tf.constant(similarity_scores)
    result = tf.nn.top_k(similarity_scores, k)
    scores = result[0]
    top_k = result[1]
    print("similarity scores:", session.run(scores))
    return session.run(top_k)


def compute_neighbors_norm(src_words2id, src_id2embedding, tar_id2embedding, src_word, k, session):
    src_id2embedding = np.array(src_id2embedding)
    src_norm = np.sqrt(np.sum(np.square(src_id2embedding), axis=1, keepdims=True))
    src_id2embedding = src_id2embedding / src_norm
    tar_id2embedding = np.array(tar_id2embedding)
    tar_norm = np.sqrt(np.sum(np.square(tar_id2embedding), axis=1, keepdims=True))
    tar_id2embedding /= tar_norm

    src_word_id = src_words2id[src_word]
    src_word = src_id2embedding[src_word_id]
    similarity_scores = np.dot(src_word, np.transpose(tar_id2embedding))

    similarity_scores = tf.constant(similarity_scores)
    result = tf.nn.top_k(similarity_scores, k)
    scores = result[0]
    top_k = result[1]
    print("similarity scores:", session.run(scores))
    return session.run(top_k)


def sample_gold_pairs(gold_dict, sample_number):
    samples = random.sample(range(len(gold_dict)), sample_number)
    new_gold_dict = []
    for sample in samples:
        new_gold_dict.append(gold_dict[sample])
    return new_gold_dict


def compute_top_k_accuracy(src_words2id, src_id2embedding, tar_words2id, tar_id2embedding, gold_dict, k, tar_id2words):
    session = tf.Session()
    true_samples = 0
    # new_gold_dict = gold_dict
    new_gold_dict = sample_gold_pairs(gold_dict, FLAGS.sample_number)
    for pair in new_gold_dict:
        src_word, tar_word = pair[0], pair[1]
        nearest_neighbors = compute_neighbors(src_words2id, src_id2embedding, tar_id2embedding, src_word, k,
                                              session)
        if tar_word not in tar_words2id:
            continue
        tar_gold_id = tar_words2id[tar_word]
        print("target words:%s" % tar_id2words[tar_gold_id])
        neighbors = "neighbors:"
        for neighbor in nearest_neighbors:
            neighbors += "%s, " % tar_id2words[neighbor]
        print(neighbors)
        print("\n")
        if tar_gold_id in nearest_neighbors:
            true_samples += 1
    accuracy = float(true_samples) / len(new_gold_dict)
    return accuracy


def main(_):
    print("read frequency words\n")
    src_freq_words = read_words_freq(FLAGS.src_freq_file)
    tar_freq_words = read_words_freq(FLAGS.tar_freq_file)

    print("filter stop words!\n")
    src_freq_words = filter_stop_words(src_freq_words, FLAGS.src_stop_words_file)
    tar_freq_words = filter_stop_words(tar_freq_words, FLAGS.tar_stop_words_file)

    print("read word embedding\n")
    src_words2id, src_id2words, src_id2embedding = read_embedding(FLAGS.src_embedding, src_freq_words)
    tar_words2id, tar_id2words, tar_id2embedding = read_embedding(FLAGS.tar_embedding, tar_freq_words)

    """
    print("filter words whose are not in embeddings\n")
    src_freq_words = filter_freq_words_embedding(src_freq_words, src_words2id)
    tar_freq_words = filter_freq_words_embedding(tar_freq_words, tar_words2id)
    """
    
    print("read gold pair")
    gold_pair = filter_gold_word_synset(FLAGS.gold_pair_file, src_freq_words, tar_freq_words)
    print("Number of Total gold pairs:%d\n" % len(gold_pair))
    print("compute accuracy")

    total_acc = []
    for i in range(10):
        acc = compute_top_k_accuracy(src_words2id, src_id2embedding, tar_words2id, tar_id2embedding, gold_pair,
                                     FLAGS.top_k, tar_id2words)
        print("sample %d: Cross lingual dictionary induction : accuracy=%f\n" % (i+1, acc))
        total_acc.append(acc)
    ave_acc = np.average(total_acc)
    max_acc = np.max(total_acc)
    print("The average accuracy=%f, the maximum accuracy=%f\n" % (ave_acc, max_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_freq_file", type=str,
                        default="E:\\NLP\\code\\bilingual_word_emb\\data\\cross-lingual dictionary induction"
                                "\\freq_words\\en-de\\en.fre.1000",
                        help="source frequency words file")
    parser.add_argument("--tar_freq_file", type=str,
                        default="E:\\NLP\\code\\bilingual_word_emb\\data\\cross-lingual dictionary induction"
                                "\\freq_words\\en-de\\de.fre.1000",
                        help="target frequency words file")
    parser.add_argument("--src_embedding", type=str,
                        default="E:\\NLP\\code\\bilingual_word_emb\\emnlp\\other_work\\BAE\\en-de\\en-de.new.en",
                        help="source embedding file")
    parser.add_argument("--tar_embedding", type=str,
                        default="E:\\NLP\\code\\bilingual_word_emb\\emnlp\\other_work\\BAE\\en-de\\en-de.new.de",
                        help="target embedding file")
    parser.add_argument("--gold_pair_file", type=str,
                        default="E:\\NLP\\code\\bilingual_word_emb\\data\\cross-lingual dictionary induction"
                                "\\gold_pair\\en-de.gold.pair",
                        help="gold pair file path")
    parser.add_argument("--src_stop_words_file", type=str,
                        default="E:\\NLP\\code\\bilingual_word_emb\\data\\cross-lingual dictionary induction\\"
                                "stop-words\\stop-word-en.txt",
                        help="file path for source stop words")
    parser.add_argument("--tar_stop_words_file", type=str,
                        default="E:\\NLP\\code\\bilingual_word_emb\\data\\cross-lingual dictionary induction\\"
                                "stop-words\\stop-word-de.txt")
    parser.add_argument("--top_k", type=int, default=1, help="number of accuracy top lists")
    parser.add_argument("--sample_number", type=int, default=100, help="number of samples for testing")

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]]+unparsed)












