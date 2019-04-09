import sys
import os
import argparse

import numpy as np
import tensorflow as tf


__author__ = "Jocelyn"

FLAGS = None


def read_freq_english_words(filename):
    src_freq_words = []
    f = open(filename, "r", encoding="utf-8")
    for line in f.readlines():
        word = line.strip()
        src_freq_words.append(word)
    return src_freq_words


def initialize_word_dict(src_freq_words):
    src2tar = dict()
    for word in src_freq_words:
        src2tar[word] = dict()
    return src2tar


def en_freq_exist(l_en, en_freq_words):
    this_en_words = l_en.strip().split()
    for word in this_en_words:
        if word in en_freq_words:
            return True
    return False


def get_aligns(l_align):
    aligns = l_align.strip().split()
    new_aligns = []
    for alignment in aligns:
        lemmas = alignment.strip().split("-")
        de_ind, en_ind = int(lemmas[0]), int(lemmas[1])
        new_aligns.append([de_ind, en_ind])
    return new_aligns


def compute_aligned_freq_de_words(src2tar):
    freq_en2de_dict = dict()
    for en_word in src2tar.keys():
        new_word_dict = sorted(src2tar[en_word].items(), key=lambda x: x[1], reverse=True)
        de_word = new_word_dict[0][0]
        freq_en2de_dict[en_word] = de_word
    return freq_en2de_dict


def write_file(de_file_name, aligned_de_words, en_freq_words):
    f = open(de_file_name, "w", encoding="utf-8")
    for word in en_freq_words:
        f.write(aligned_de_words[word])
        f.write("\n")
    f.close()


def get_aligned_words(de_file, en_file, align_file, src2tar, en_freq_words, write_file_name):
    f_de = open(de_file, "r", encoding="utf-8")
    f_en = open(en_file, "r", encoding="utf-8")
    align_file = open(align_file, "r", encoding="utf-8")
    for l_de, l_en, l_align in zip(f_de.readlines(), f_en.readlines(), align_file.readlines()):
        if not en_freq_exist(l_en, en_freq_words):
            continue
        de_words = l_de.strip().split()
        en_words = l_en.strip().split()
        aligns = get_aligns(l_align)

        for alignment in aligns:
            de_ind, en_ind = alignment[0], alignment[1]
            de_word, en_word = de_words[de_ind], en_words[en_ind]
            if en_word in src2tar:
                if de_word in src2tar[en_word]:
                    src2tar[en_word][de_word] += 1
                else:
                    src2tar[en_word][de_word] = 1

    aligned_de_words = compute_aligned_freq_de_words(src2tar)
    write_file(write_file_name, aligned_de_words, en_freq_words)


def main(_):
    en_freq_words = read_freq_english_words(FLAGS.en_freq_words)
    en2de_dict = initialize_word_dict(en_freq_words)

    get_aligned_words(FLAGS.de_file, FLAGS.en_file, FLAGS.align_file, en2de_dict, en_freq_words, FLAGS.write_file_name)


if __name__ == "__main__":
    path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath("compute_aligned_words.py"))))
    data_path = os.path.join(path, "data\\de-en")

    parse = argparse.ArgumentParser()
    parse.add_argument("--de_file", type=str, default=os.path.join(data_path, "result\\result\\de-en.de"),
                       help="German file path")
    parse.add_argument("--en_file", type=str, default=os.path.join(data_path, "result\\result\\de-en.en"),
                       help="English file path")
    parse.add_argument("--align_file", type=str,
                       default=os.path.join(data_path, "result\\result\\model\\aligned.grow-diag-final-and"),
                       help="alignment file path")
    parse.add_argument("--en_freq_words", type=str, default=os.path.join(data_path, "visualize\\en.freq.words.txt"),
                       help="English frequent words file!")
    parse.add_argument("--write_file_name", type=str, default=os.path.join(data_path, "visualize\\en.freq.de.txt"))

    FLAGS, unparsed = parse.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]]+unparsed)

