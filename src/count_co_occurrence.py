import os
import numpy as np
import argparse


def add_line_word_count(word_dict, words):
    for word in words:
        if word not in word_dict.keys():
            word_dict[word] = 0
        word_dict[word] += 1
    return word_dict


def add_line_cross_word_count(cross_word_dict, src_words, tar_words):
    for src_word in src_words:
        if src_word not in cross_word_dict.keys():
            cross_word_dict[src_word] = dict()
        for tar_word in tar_words:
            if tar_word not in cross_word_dict[src_word].keys():
                cross_word_dict[src_word][tar_word] = 0
            cross_word_dict[src_word][tar_word] += 1
    return cross_word_dict


def compute_word_count(src_file, tar_file):
    f_src = open(src_file, "r", encoding="utf-8")
    f_tar = open(tar_file, "r", encoding="utf-8")

    src_dict, tar_dict = dict(), dict()
    src2tar_dict, tar2src_dict = dict(), dict()

    index = 0
    for src_line, tar_line in zip(f_src.readlines(), f_tar.readlines()):
        index += 1
        print(index)

        src_words = set(src_line.strip().split())
        tar_words = set(tar_line.strip().split())

        src_dict = add_line_word_count(src_dict, src_words)
        tar_dict = add_line_word_count(tar_dict, tar_words)

        src2tar_dict = add_line_cross_word_count(src2tar_dict, src_words, tar_words)
        tar2src_dict = add_line_cross_word_count(tar2src_dict, tar_words, src_words)

    return src_dict, tar_dict, src2tar_dict, tar2src_dict


def compute_align_weight_score(word_dict, cross_word_dict):
    """
    :param word_dict:
    :param cross_word_dict:
    :return: {"me": [("他", 1), ("我", 2)]}
    """
    for src_word in cross_word_dict.keys():
        word_count = word_dict[src_word]

        for tar_word in cross_word_dict[src_word].keys():
            cross_word_dict[src_word][tar_word] = float(cross_word_dict[src_word][tar_word]) / float(word_count)

        cross_word_dict[src_word] = sorted(cross_word_dict[src_word].items(), key=lambda d: d[1], reverse=True)

    return cross_word_dict


def write_cross_dict_weight_score(cross_word_dict, filename):
    f = open(filename, "w", encoding="utf-8")

    for src_word in cross_word_dict.keys():
        word_dict = cross_word_dict[src_word]
        for tar_word, score in word_dict:
            line = src_word + "|||" + tar_word + "|||" + str(score) + "\n"
            f.write(line)
    f.close()


def compute_idf_score(filename):
    idf_dict = dict()
    f = open(filename, "r", encoding="utf-8")
    lines = f.readlines()
    for line in lines:
        words = set(line.strip().split())
        for word in words:
            if word not in idf_dict:
                idf_dict[word] = 0
            idf_dict[word] += 1
    doc = len(lines)
    f.close()
    for word in idf_dict.keys():
        idf_dict[word] = np.log(float(doc) / idf_dict[word])
    return idf_dict


def write_idf_score(idf_dict, filename):
    f = open(filename, "w", encoding="utf-8")
    for word in idf_dict.keys():
        line = word + "|||" + str(idf_dict[word]) + "\n"
        f.write(line)
    f.close()


def add_cross_word_to_dict(cross_word_dict, src_word, tar_word, score):
    if src_word not in cross_word_dict.keys():
        cross_word_dict[src_word] = dict()
    cross_word_dict[src_word][tar_word] = score
    return cross_word_dict


def read_cross_word_score(filename):
    cross_word_dict = dict()
    f = open(filename, "r", encoding="utf-8")
    for line in f.readlines():
        if "||||" in line:
            continue
        lemmas = line.strip().split("|||")
        src_word = lemmas[0].strip()
        tar_word = lemmas[1].strip()
        score = float(lemmas[2].strip())
        cross_word_dict = add_cross_word_to_dict(cross_word_dict, src_word, tar_word, score)
    return cross_word_dict


def add_word_to_idf_dict(idf_dict, word, score):
    idf_dict[word] = score
    return idf_dict


def read_idf_score(filename):
    idf_dict = dict()
    f = open(filename, "r", encoding="utf-8")
    for line in f.readlines():
        if "||||" in line:
            continue
        lemmas = line.strip().split("|||")
        word = lemmas[0].strip()
        score = float(lemmas[1].strip())
        idf_dict = add_word_to_idf_dict(idf_dict, word, score)
    return idf_dict


def normalize_word_count(word_dict):
    total_value = sum(word_dict.values())
    for word in word_dict.keys():
        word_dict[word] = word_dict[word] / total_value
    return word_dict


def compute_cross_word_add_idf_score(cross_word_dict, tar_idf_score):
    for src_word in cross_word_dict.keys():
        for tar_word in cross_word_dict[src_word].keys():
            cross_word_dict[src_word][tar_word] = cross_word_dict[src_word][tar_word] * tar_idf_score[tar_word]
        cross_word_dict[src_word] = normalize_word_count(cross_word_dict[src_word])
        cross_word_dict[src_word] = sorted(cross_word_dict[src_word].items(), key=lambda d: d[1], reverse=True)
    return cross_word_dict


if __name__ == "__main__":
    dir_path = "E:\\NLP\\code\\bilingual_word_emb\\code\\sent_based_bwe\\data\\en-fr"

    """
    # compute word co-occurrence score
    print("compute co-occurrence\n")
    de_dict, en_dict, de2en_dict, en2de_dict = compute_word_count(os.path.join(dir_path,
                                                                               "fr-en.true.fr"),
                                                                  os.path.join(dir_path,
                                                                               "fr-en.true.en"))

    print("Normalize align score\n")
    # de2en_dict = compute_align_weight_score(de_dict, de2en_dict)
    # en2de_dict = compute_align_weight_score(en_dict, en2de_dict)

    print("Write align score\n")
    # write_cross_dict_weight_score(de2en_dict, os.path.join(dir_path, "word_idf_count\\fr2en.word.count"))
    # write_cross_dict_weight_score(en2de_dict, os.path.join(dir_path, "word_idf_count\\en2fr.word.count"))

    # compute idf scores
    print("compute idf scores\n")
    de_idf_score_dict = compute_idf_score(os.path.join(dir_path, "fr-en.true.fr"))
    en_idf_score_dict = compute_idf_score(os.path.join(dir_path, "fr-en.true.en"))

    print("write idf scores to file\n")
    write_idf_score(de_idf_score_dict, os.path.join(dir_path, "word_idf_count\\fr.idf.score"))
    write_idf_score(en_idf_score_dict, os.path.join(dir_path, "word_idf_count\\en.idf.score"))

    """
    # read cross word score
    print("read cross word dict!\n")
    de2en_word_dict = read_cross_word_score(os.path.join(dir_path, "word_idf_count\\fr2en.word.count"))
    en2de_word_dict = read_cross_word_score(os.path.join(dir_path, "word_idf_count\\en2fr.word.count"))

    print("read idf score\n")
    de_idf_dict = read_idf_score(os.path.join(dir_path, "word_idf_count\\fr.idf.score"))
    en_idf_dict = read_idf_score(os.path.join(dir_path, "word_idf_count\\en.idf.score"))

    print("compute word idf added count score\n")
    zh2en_word_dict = compute_cross_word_add_idf_score(de2en_word_dict, en_idf_dict)
    en2zh_word_dict = compute_cross_word_add_idf_score(en2de_word_dict, de_idf_dict)

    print("write score\n")
    write_cross_dict_weight_score(zh2en_word_dict,
                                  os.path.join(dir_path, "word_idf_count\\fr2en.word.idf.count"))
    write_cross_dict_weight_score(en2zh_word_dict,
                                  os.path.join(dir_path, "word_idf_count\\en2fr.word.idf.count"))














