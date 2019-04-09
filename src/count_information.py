import os

import  numpy


def read_line_words(words, word_count_dict, cross_word_count_dict):
    for word in words:
        if word in word_count_dict:
            word_count_dict[word] += 1
        else:
            word_count_dict[word] = 1
        if word not in cross_word_count_dict:
            cross_word_count_dict[word] = {}
    return word_count_dict, cross_word_count_dict


def count_word_information(zh_file, en_file):
    f1 = open(zh_file, "r", encoding="utf-8")
    f2 = open(en_file, "r", encoding="utf-8")
    zh_word_count_dict, en_word_count_dict = {}, {}
    zh2en_count_dict, en2zh_count_dict = {}, {}
    index = 0
    for zh_line, en_line in zip(f1.readlines(), f2.readlines()):
        index += 1
        print(index)
        zh_words = zh_line.strip().split()
        en_words = en_line.strip().split()

        zh_word_count_dict, zh2en_count_dict = read_line_words(zh_words, zh_word_count_dict, zh2en_count_dict)
        en_word_count_dict, en2zh_count_dict = read_line_words(en_words, en_word_count_dict, en2zh_count_dict)

        for zh_word in zh_words:
            for en_word in en_words:
                if en_word in zh2en_count_dict[zh_word]:
                    zh2en_count_dict[zh_word][en_word] += 1
                else:
                    zh2en_count_dict[zh_word][en_word] = 1

                if zh_word in en2zh_count_dict[en_word]:
                    en2zh_count_dict[en_word][zh_word] += 1
                else:
                    en2zh_count_dict[en_word][zh_word] = 1
    return zh_word_count_dict, en_word_count_dict, zh2en_count_dict, en2zh_count_dict


def save_count_file(word_count_file, word_count_dict):
    f = open(word_count_file, "w", encoding="utf-8")
    for word in word_count_dict.keys():
        s = word + "|||" + str(word_count_dict[word]) + "\n"
        f.write(s)
    f.close()


def save_cross_word_count_file(cross_word_count_file, cross_word_count_dict):
    f = open(cross_word_count_file, "w", encoding="utf-8")
    for word in cross_word_count_dict.keys():
        word_dict = cross_word_count_dict[word]
        for cross_word in word_dict.keys():
            s = word + "|||" + cross_word + "|||" + str(word_dict[cross_word]) + "\n"
            f.write(s)
    f.close()


if __name__ == "__main__":
    path = os.path.dirname(os.path.dirname(os.path.abspath("count_information.py")))
    data_path = os.path.join(path, "data\\valid\\tune")

    zh_word_count, en_word_count, zh2en_count, en2zh_count = \
        count_word_information(os.path.join(data_path, "zh.raw"),
                               os.path.join(data_path, "en.raw"))

    save_count_file(os.path.join(data_path, "zh.word.count"),
                    zh_word_count)
    save_count_file(os.path.join(data_path, "en.word.count"),
                    en_word_count)
    save_cross_word_count_file(os.path.join(data_path, "zh.cross.word.count"),
                               zh2en_count)
    save_cross_word_count_file(os.path.join(data_path, "en.cross.word.count"),
                               en2zh_count)





