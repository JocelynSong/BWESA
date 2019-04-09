import collections
import random

import tensorflow as tf
import numpy as np

from src.data_utils import UNK

__author__ = "Jocelyn"


def read_data(filename):
    f = open(filename, "r", encoding="utf-8")
    data = []
    for line in f.readlines():
        data.extend(line.strip().split())
    return data


def build_dataset(data, min_count):
    """
    get word2id(word, count) and id2word(id, word)
    :param data:
    :param min_count:
    :return:
    """
    count = [[UNK, -1]]
    count.extend(collections.Counter(data).most_common(len(data)))
    word2id = dict()
    word2id[UNK] = 0
    id2count = dict()
    id2count[0] = -1
    for word, number in count:
        if number >= min_count:
            word2id[word] = len(word2id)
            id2count[word2id[word]] = number
        else:
            id2count[word2id[UNK]] += 1
    # id2word = dict(zip(word2id.values(), word2id.keys()))
    id2word = {value: key for key, value in word2id.items()}
    return word2id, id2count, id2word


def change_contexts_to_indexs(data, word2id):
    new_data = [word2id[word] if word in word2id else word2id[UNK] for word in data]
    return new_data


def save_vocab(word2id, id2count, savefile):
    f = open(savefile, "w", encoding="utf-8")
    for word, ind in word2id.items():
        f.write(word + "|||" + str(id2count[ind]) + "\n")
    f.close()


def get_word_id(word2id, word):
    if word in word2id:
        ind = word2id[word]
    else:
        ind = word2id[UNK]
    return ind


def word_skip_gram(filename, train_words, word2id, id2count, window_size, sub_sample):
    f = open(filename, "r", encoding="utf-8")
    examples = []
    labels = []
    for line in f.readlines():
        words = line.strip().split()
        for i in range(len(words)):
            word = words[i]
            ind = get_word_id(word2id, word)
            # ind = word2id[word]
            count = id2count[ind]
            ran = (np.sqrt(count / (sub_sample * train_words)) + 1) * (sub_sample * train_words) / count
            rand = random.random()
            if ran < rand:
                continue
            start = i - window_size
            end = i + window_size
            while start <= end:
                if start == i:
                    start += 1
                    continue
                elif start < 0:
                    start += 1
                    continue
                elif start >= len(words):
                    break
                else:
                    examples.append(ind)
                    label_ind = get_word_id(word2id, words[start])
                    labels.append(label_ind)
                    # labels.append(word2id[words[start]])
                    start += 1
    f.close()
    return examples, labels


def align_batch_size(examples, labels, batch_size):
    if len(examples) % batch_size != 0:
        remain = batch_size - len(examples) % batch_size
        for i in range(remain):
            ind = random.randint(0, len(examples)-1)
            examples.append(examples[ind])
            labels.append(labels[ind])
    return examples, labels


def sort_parallel_sentence(train_sen, max_len, src_word2id, tar_word2id):
    """
    sort parallel sentence according to source sentence length and return source and target sentence indexs and length
    src_sen:[[1, 2, 3], [3, 2, 9]]
    src_len:[3, 3]
    :param train_sen:
    :param max_len:
    :param src_word2id:
    :param tar_word2id:
    :return:
    """
    train_sen = sorted(train_sen, key=lambda d: len(d[0]))

    src_sen = [sen[0] for sen in train_sen]
    src_len = [len(sen) for sen in src_sen]
    src_sen = align_sentence_length(src_sen, max_len, src_word2id)

    tar_sen = [sen[1] for sen in train_sen]
    tar_len = [len(sen) for sen in tar_sen]
    tar_sen = align_sentence_length(tar_sen, max_len, tar_word2id)
    return src_sen, src_len, tar_sen, tar_len


def read_parallel_sentence(src_file, tar_file, src_word2id, tar_word2id):
    """
    read source file and target file to get our own parallel sentence pair
    filter sentences whose length are less than 20 or more than 50
    :param src_file:
    :param tar_file:
    :param src_word2id:
    :param tar_word2id:
    :return:
    """
    f_src = open(src_file, "r", encoding="utf-8")
    f_tar = open(tar_file, "r", encoding="utf-8")
    src_lines = f_src.readlines()
    tar_lines = f_tar.readlines()
    train_sen = list()
    for src, tar in zip(src_lines, tar_lines):
        src_words = src.strip().split()
        if len(src_words) < 20 or len(src_words) > 50:
            continue
        tar_words = tar.strip().split()
        if len(tar_words) < 20 or len(tar_words) > 50:
            continue
        src_indexes = [src_word2id[word] if word in src_word2id else src_word2id[UNK] for word in src_words]
        tar_indexes = [tar_word2id[word] if word in tar_word2id else tar_word2id[UNK] for word in tar_words]
        train_sen.append([src_indexes, tar_indexes])
    f_src.close()
    f_tar.close()
    return train_sen


def get_sentence_buckets():
    sen_buckets = dict()
    sen_buckets[0] = list()
    sen_buckets[1] = list()
    sen_buckets[2] = list()
    sen_buckets[3] = list()
    return sen_buckets


def split_buckets_sentence(train_sen):
    train_sen = sorted(train_sen, key=lambda d: (len(d[0]), len(d[1])))

    buckets = [25, 50, 75, 100]
    train_sen_buckets = get_sentence_buckets()

    for sen in train_sen:
        src_sen, tar_sen = sen[0], sen[1]
        ind = 0
        while not(len(src_sen) <= buckets[ind] and len(tar_sen) <= buckets[ind]):
            ind += 1
            if ind > 3:
                break
        if ind > 3:
            continue
        train_sen_buckets[ind].append(sen)
    return train_sen_buckets


def align_sen_batch(train_sen, batch_size):
    if len(train_sen) % batch_size == 0:
        return train_sen
    number = batch_size - len(train_sen) % batch_size
    indexs = range(len(train_sen))
    for i in range(number):
        ind = random.choice(indexs)
        train_sen.append(train_sen[ind])
    return train_sen


def align_buckets_sentence_batch(train_sen_buckets, batch_size):
    for key in train_sen_buckets.keys():
        train_sen_buckets[key] = align_sen_batch(train_sen_buckets[key], batch_size)
    return train_sen_buckets


def align_sentence_length(sen_data, max_len, word2id):
    unk_ind = word2id[UNK]
    new_sen_data = []
    for sen in sen_data:
        if len(sen) < max_len:
            for _ in range(max_len - len(sen)):
                sen.append(unk_ind)
        else:
            sen = sen[:max_len]
        new_sen_data.append(sen)
    return new_sen_data


def align_one_sentence_length(sen, max_len, word2id):
    unk_ind = word2id[UNK]
    if len(sen) < max_len:
        for _ in range(max_len - len(sen)):
            sen.append(unk_ind)
    else:
        sen = sen[:max_len]
    return sen


def align_buckets_sentence_length(train_sen_buckets, src_word2id, tar_word2id):
    buckets = [25, 50, 75, 100]
    for key in train_sen_buckets.keys():
        bucket_sen = train_sen_buckets[key]
        new_bucket_sen = []
        for sen in bucket_sen:
            src_sen, tar_sen = sen[0], sen[1]
            new_src_sen = align_one_sentence_length(src_sen, buckets[key], src_word2id)
            new_tar_sen = align_one_sentence_length(tar_sen, buckets[key], tar_word2id)
            new_bucket_sen.append([new_src_sen, new_tar_sen])
        train_sen_buckets[key] = new_bucket_sen
    return train_sen_buckets


def read_cross_word_weight(filename, src_word2id, tar_word2id):
    cross_word_dict = dict()
    f = open(filename, "r", encoding="utf-8")
    for line in f.readlines():
        lemmas = line.strip().split("|||")
        src_word = lemmas[0].strip()
        tar_word = lemmas[1].strip()
        score = float(lemmas[2].strip())
        if not(src_word in src_word2id.keys() and tar_word in tar_word2id.keys()):
            continue
        src_word_id = src_word2id[src_word]
        tar_word_id = tar_word2id[tar_word]
        if src_word_id not in cross_word_dict.keys():
            cross_word_dict[src_word_id] = dict()
        cross_word_dict[src_word_id][tar_word_id] = score
    f.close()
    return cross_word_dict


def get_top_k_cross_word(word_dict, k):
    word_dict = sorted(word_dict.items(), key=lambda d: d[1], reverse=True)
    new_word_dict = dict()
    for tar_word, score in word_dict:
        if len(new_word_dict.keys()) >= k:
            break
        new_word_dict[tar_word] = score
    return new_word_dict


def filter_cross_word_dict(cross_word_dict, k):
    for src_word in cross_word_dict.keys():
        word_dict = cross_word_dict[src_word]
        if len(word_dict.keys()) <= k:
            continue
        new_word_dict = get_top_k_cross_word(word_dict, k)
        cross_word_dict[src_word] = new_word_dict
    return cross_word_dict















































