import os
import time
import sys

import tensorflow as tf
import numpy as np
import random

__author__ = "Jocelyn"


def read_file(filename):
    """
    read source, target and alignment files
    :param filename:
    :return:
    """
    f = open(filename, "r", encoding="utf-8")
    lines = f.readlines()
    return lines


def compute_sentence_length(lines):
    number = 0
    for line in lines:
        words = line.strip().split()
        number += len(words)
    number = float(number) / len(lines)
    return number


def sample_examples(total_num, sample_number):
    """
    sample examples from original data
    :param total_num:
    :param sample_number:
    :return:
    """
    samples = random.sample(range(total_num), sample_number)
    return samples


def get_samples(src_lines, tar_lines, align_lines, sample_number):
    samples = sample_examples(len(src_lines), sample_number)

    src_samples, tar_samples, align_samples = [], [], []
    for ind in samples:
        src_samples.append(src_lines[ind])
        tar_samples.append(tar_lines[ind])
        align_samples.append(align_lines[ind])
    return src_samples, tar_samples, align_samples


def get_word_embedding(line):
    lemmas = line.strip().split()
    word = lemmas[0].strip()
    embedding = []
    for lemma in lemmas[1:]:
        embedding.append(float(lemma.strip()))
    return word, embedding


def read_word_embedding(filename):
    f = open(filename, "r", encoding="utf-8")
    embeddings = []
    word2id = {}
    for line in f.readlines():
        word, embedding = get_word_embedding(line)
        embeddings.append(embedding)
        word2id[word] = len(word2id)
    id2word = {index: word for word, index in word2id.items()}
    return word2id, id2word, embeddings


def get_sen_words(sen):
    words = sen.strip().split()
    return words


def get_alignment_index(align):
    inds = align.strip().split("-")
    src_ind = int(inds[1].strip())
    tar_ind = int(inds[0].strip())
    return src_ind, tar_ind


def get_align_distribution(alignment, src_len, tar_len):
    score = np.array(np.zeros([src_len, tar_len]), dtype=float)
    aligns = alignment.strip().split()
    for align in aligns:
        src_ind, tar_ind = get_alignment_index(align)
        score[src_ind][tar_ind] = 1.0
    for i in range(src_len):
        for j in range(tar_len):
            score += 1.0
    align_sum = np.sum(score, axis=1, keepdims=True)
    score /= align_sum
    return score


def compute_true_distribution(src_sen, tar_sen, align):
    """
    compute s_ij, the true distribution
    :param src_sen:
    :param tar_sen:
    :param align:
    :return:
    """
    src_words = get_sen_words(src_sen)
    tar_words = get_sen_words(tar_sen)
    align_dis = get_align_distribution(align, len(src_words), len(tar_words))
    return align_dis


def relu(x):
    if x > 0:
        return x
    else:
        return 0


def compute_proximate_distribution(src_sen, tar_sen, src_word2id, src_embeddings, tar_word2id, tar_embeddings):
    """
    compute align score based on dot product
    :param src_sen:
    :param tar_sen:
    :param src_word2id:
    :param src_embeddings:
    :param tar_word2id:
    :param tar_embeddings:
    :return:
    """
    src_words = get_sen_words(src_sen)
    tar_words = get_sen_words(tar_sen)
    pro_score = np.array(np.zeros([len(src_words), len(tar_words)]), dtype=float)

    for i in range(len(src_words)):
        for j in range(len(tar_words)):
            src = src_words[i]
            tar = tar_words[j]

            if src in src_word2id:
                src_embedding = src_embeddings[src_word2id[src]]
            else:
                src_embedding = src_embeddings[src_word2id['<UNK>']]
            if tar in tar_word2id:
                tar_embedding = tar_embeddings[tar_word2id[tar]]
            else:
                tar_embedding = tar_embeddings[tar_word2id["<UNK>"]]
            pro_score[i][j] = relu(np.dot(src_embedding, tar_embedding))
    pro_sum = np.sum(pro_score, axis=1, keepdims=True)
    pro_score = pro_score / pro_sum
    return pro_score


def compute_kl_dist_per_word(true_dis, pro_dis, ind):
    dist = 0
    for s, a in zip(true_dis[ind], pro_dis[ind]):
        dist += s * np.log(s / a)
    return dist


def compute_kl_dist_per_sen(true_distribution, proximate_distribution):
    dist = 0
    for i in range(len(true_distribution)):
        kl_dist_word = compute_kl_dist_per_word(true_distribution, proximate_distribution, i)
        dist += kl_dist_word
    # dist /= len(true_distribution)
    return dist


def compute_kl_sen(src_sen, tar_sen, align, src_word2id, src_embeddings, tar_word2id, tar_embeddings):
    """
    compute KL distance
    :param src_sen:
    :param tar_sen:
    :param align:
    :param src_word2id:
    :param src_embeddings:
    :param tar_word2id:
    :param tar_embeddings:
    :return:
    """
    true_distribution = compute_true_distribution(src_sen, tar_sen, align)
    proximate_distribution = compute_proximate_distribution(src_sen, tar_sen, src_word2id, src_embeddings, tar_word2id,
                                                            tar_embeddings)

    kl_dist_sen = compute_kl_dist_per_sen(true_distribution, proximate_distribution)
    return kl_dist_sen


def compute_kl_dist(src_sens, tar_sens, aligns, src_word2id, src_embeddings, tar_word2id, tar_embeddings):
    dist = 0
    for src_sen, tar_sen, align in zip(src_sens, tar_sens, aligns):
        dist += compute_kl_sen(src_sen, tar_sen, align, src_word2id, src_embeddings, tar_word2id, tar_embeddings)
    dist /= len(src_sens)
    return dist


def compute_kl_sen_reverse(src_sen, tar_sen, align, src_word2id, src_embeddings, tar_word2id, tar_embeddings):
    """
    compute KL distance
    :param src_sen:
    :param tar_sen:
    :param align:
    :param src_word2id:
    :param src_embeddings:
    :param tar_word2id:
    :param tar_embeddings:
    :return:
    """
    true_distribution = compute_true_distribution(tar_sen, src_sen, align)
    proximate_distribution = compute_proximate_distribution(tar_sen, src_sen, tar_word2id, tar_embeddings, src_word2id,
                                                            src_embeddings)

    kl_dist_sen = compute_kl_dist_per_sen(true_distribution, proximate_distribution)
    return kl_dist_sen


def compute_kl_dist_reverse(src_sens, tar_sens, aligns, src_word2id, src_embeddings, tar_word2id, tar_embeddings):
    dist = 0
    for src_sen, tar_sen, align in zip(src_sens, tar_sens, aligns):
        dist += compute_kl_sen_reverse(src_sen, tar_sen, align, src_word2id, src_embeddings, tar_word2id,
                                       tar_embeddings)
    dist /= len(src_sens)
    return dist


def get_word_level_alignment(score, ind):
    num = 0
    for s in score[ind]:
        if s - 0.06 > 0:
            num += 1
    return num


def compute_averaged_align_words(score):
    num = 0
    for i in range(len(score)):
        this_num = get_word_level_alignment(score, i)
        num += this_num
    num = float(num) / len(score)
    return num


def compute_aligned_words(src_sen, tar_sen, src_word2id, src_embeddings, tar_word2id, tar_embeddings):
    """
    compute align score based on dot product
    :param src_sen:
    :param tar_sen:
    :param src_word2id:
    :param src_embeddings:
    :param tar_word2id:
    :param tar_embeddings:
    :return:
    """
    src_words = get_sen_words(src_sen)
    tar_words = get_sen_words(tar_sen)
    pro_score = np.array(np.zeros([len(src_words), len(tar_words)]), dtype=float)

    for i in range(len(src_words)):
        for j in range(len(tar_words)):
            src = src_words[i]
            tar = tar_words[j]

            if src in src_word2id and tar in tar_word2id:
                src_embedding = src_embeddings[src_word2id[src]]
                tar_embedding = tar_embeddings[tar_word2id[tar]]
                pro_score[i][j] = relu(np.dot(src_embedding, tar_embedding))
            else:
                pro_score[i][j] = 0.
    pro_sum = np.sum(pro_score, axis=1, keepdims=True)
    pro_score = pro_score / pro_sum
    print(pro_score)
    number = compute_averaged_align_words(pro_score)
    return number


def compute_examples_aligned_words(src_samples, tar_samples, src_word2id, src_embeddings, tar_word2id, tar_embeddings):
    number = 0
    for src_sen, tar_sen in zip(src_samples, tar_samples):
        this_num = compute_aligned_words(src_sen, tar_sen, src_word2id, src_embeddings, tar_word2id, tar_embeddings)
        number += this_num
    number = number / len(src_samples)
    return number


def main(_):
    dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath("compute_kl_distance.py"))))
    data_path = os.path.join(dir_path, "data\\de-en")
    emb_path = os.path.join(data_path, "embedding")

    src_file = os.path.join(data_path, "europarl-v7.de-en.de.tok.true")
    tar_file = os.path.join(data_path, "europarl-v7.de-en.en.tok.true")
    align_file = os.path.join(data_path, "result\\result\\model\\aligned.grow-diag-final-and")

    src_lines = read_file(src_file)
    tar_lines = read_file(tar_file)
    # align_lines = read_file(align_file)
    # src_samples, tar_samples, align_samples = get_samples(src_lines, tar_lines, align_lines, 1000)

    zh_length = compute_sentence_length(src_lines)
    en_length = compute_sentence_length(tar_lines)
    print("averaged sentence length:de=%f, en=%f\n" % (zh_length, en_length))

    """
    # baseline
    # src_emb_file = os.path.join(emb_path, "de.emb.de_en_dimension40_alpha0.60_beta0.40_gama1.00_lambda4.00_epoch25.vec")
    # tar_emb_file = os.path.join(emb_path, "en.emb.de_en_dimension40_alpha0.60_beta0.40_gama1.00_lambda4.00_epoch25.vec")
    # our model
    src_emb_file = os.path.join(emb_path, "zh.emb.dimension300_alpha0.50_beta0.50_gama1.00_lambda8.00_epoch19.vec")
    tar_emb_file = os.path.join(emb_path, "en.emb.dimension300_alpha0.50_beta0.50_gama1.00_lambda8.00_epoch19.vec")
    src_word2id, src_id2word, src_embeddings = read_word_embedding(src_emb_file)
    tar_word2id, tar_id2word, tar_embeddings = read_word_embedding(tar_emb_file)

    ave_num = compute_examples_aligned_words(src_samples, tar_samples, src_word2id, src_embeddings, tar_word2id,
                                             tar_embeddings)
    # ave_num = compute_examples_aligned_words(tar_samples, src_samples, tar_word2id, tar_embeddings, src_word2id,
    #                                         src_embeddings)
    print("The averaged number of aligned words per word is: %f\n" % ave_num)

    
    # from source to target
    dist = compute_kl_dist(src_samples, tar_samples, align_samples, src_word2id, src_embeddings, tar_word2id,
                           tar_embeddings)
    print("The KL distance is: %f\n" % dist)

    
    # from target to source
    # switch between src2tar and tar2src, the alignment should be changed , and others shouldn't be changed
    dist = compute_kl_dist_reverse(src_samples, tar_samples, align_samples, src_word2id, src_embeddings, tar_word2id,
                                   tar_embeddings)
    print("The tar to src distance is: %f\n" % dist)
    """


if __name__ == "__main__":
    tf.app.run(main=main, argv=[sys.argv[0]])










