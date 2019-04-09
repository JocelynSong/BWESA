# -*- coding: UTF-8 -*-
import os
import codecs
import random
from nltk.tokenize.stanford_segmenter import StanfordSegmenter


def get_text(filename):
    f = codecs.open(filename, "r", encoding="utf-8")
    lines = f.readlines()
    start_line = 0
    end_line = 0
    for i in range(len(lines)):
        if "<text>" in lines[i]:
            start_line = i
        if "</text>" in lines[i]:
            end_line = i
    text = ""

    lines = lines[start_line + 1: end_line]
    for line in lines:
        text += line
    text = text.replace("<p>", "")
    text = text.replace("</p>", "")
    text = text.replace("&quot;", "\"")
    return text


def get_labels(filename):
    """
     CCAT（0）, ECAT（1）, GCAT（2）, MCAT（3）
    :param filename:
    :return:
    """
    f = open(filename, "r", encoding="utf-8")
    labels = []
    for line in f.readlines():
        if "<code code=\"CCAT\">" in line:
            labels.append(0)
        elif "<code code=\"ECAT\">" in line:
            labels.append(1)
        elif "<code code=\"GCAT\">" in line:
            labels.append(2)
        elif "<code code=\"MCAT\">" in line:
            labels.append(3)
        else:
            continue
    return labels


def get_information(filename):
    text = get_text(filename)
    labels = get_labels(filename)
    return labels, text


def write_file(filename, labels, text):
    f = open(filename, "w", encoding="utf-8")
    line = ""
    if len(labels) == 0:
        line += "-1"
    elif len(labels) == 1:
        line += str(labels[0])
    else:
        line += str(labels[0])
        for label in labels[1:]:
            line = line + "|||" + str(label)
    line += "\n"
    f.write(line)
    f.write(text)
    f.close()


def get_documents_done_with_label(dir_path, dir_path2):
    dirs = os.listdir(dir_path)
    new_dir_path = dir_path2
    os.mkdir(new_dir_path)
    index = 0
    for directory in dirs:
        print(str(index) + directory)
        index += 1
        new_child_dir_path = os.path.join(new_dir_path, directory)
        os.mkdir(new_child_dir_path)
        child_dir = os.path.join(dir_path, directory)
        files = os.listdir(child_dir)
        for file in files:
            filename = os.path.join(child_dir, file)
            labels, text = get_information(filename)
            print(labels)
            print(text)
            filename2 = os.path.join(new_child_dir_path, file.replace(".xml", "")) + ".txt"
            write_file(filename2, labels, text)


def test(filename, filename2):
    labels, text = get_information(filename)
    write_file(filename2, labels, text)


def is_ok_to_keep(start_file):
    f = open(start_file, "r", encoding="utf-8")
    indexs = f.readline().strip().split("|||")
    if len(indexs) == 1 and int(indexs[0].strip()) != -1:
        return True
    return False


def write_single_label_file(start_file, end_file):
    f1 = open(start_file, "r", encoding="utf-8")
    f2 = open(end_file, "w", encoding="utf-8")
    for line in f1.readlines():
        f2.write(line)
        print(line)
    f1.close()
    f2.close()


def get_single_label_english_file(dir_path1, dir_path2):
    dirs = os.listdir(dir_path1)
    for directory in dirs:
        child_dir = dir_path1 + "\\" + directory
        files = os.listdir(child_dir)
        for file in files:
            start_file = dir_path1 + "\\" + directory + "\\" + file
            end_file = dir_path2 + "\\" + file
            if is_ok_to_keep(start_file):
                print("ok")
                write_single_label_file(start_file, end_file)


def sample_documents(dir_path1, dir_path2, sample_number):
    files = os.listdir(dir_path1)
    sample_number = max(sample_number, len(files))
    print("sample number:%d\n" % len(files))
    samples = random.sample(range(len(files)), sample_number)
    for index in samples:
        start_file = os.path.join(dir_path1, files[index])
        end_file = os.path.join(dir_path2, files[index])
        write_single_label_file(start_file, end_file)


def count_words_of_file(filename):
    f = open(filename, "r", encoding="utf-8")
    number = 0
    for line in f.readlines():
        words = line.strip().split()
        number += len(words)
    return number


def count_words_number(dir_path):
    count = 0
    files = os.listdir(dir_path)
    for file in files:
        filename = os.path.join(dir_path, file)
        count += count_words_of_file(filename)
    return count


def segment_words(dir_path1, dir_path2):
    files = os.listdir(dir_path1)
    seg = StanfordSegmenter(path_to_jar="E:\\class\\statistical method\\project\\pj1\\stanford-segmenter-2017-06-09"
                                        "\\stanford-segmenter-2017-06-09\\stanford-segmenter-3.8.0.jar",
                            path_to_model="E:\\class\\statistical method\\project\\pj1\\stanford-segmenter-2017-06-09\\"
                                          "stanford-segmenter-2017-06-09\\data\\arabic-segmenter-atb+bn+arztrain.ser.gz")
    seg.default_config("ar")
    for file in files:
        filename = os.path.join(dir_path1, file)
        filename2 = os.path.join(dir_path2, file)
        if os.path.exists(filename2):
            continue
        f1 = open(filename, "r", encoding="utf-8")
        f2 = open(filename2, "w", encoding="utf-8")
        new_content = seg.segment_file(filename)
        print(new_content)
        f2.write(new_content)
        """
        for line in f1.readlines():
            seg_line = seg.segment(line.strip())
            f2.write(seg_line)
            f2.write("\n")
        """
        f1.close()
        f2.close()


if __name__ == "__main__":
    """
    split_char = [",", ".", ":", "\"", "\'", ";", ")", "(", "?"]

    dir_path1 = "E:\\NLP\\code\\bilingual_word_emb\\data\\cross-lingual document classification\\rcv2\\" \
                "RCV2_sample\\german"
    dir_path2 = "E:\\NLP\\code\\bilingual_word_emb\\data\\cross-lingual document classification\\rcv2\\" \
                "RCV2_sample_seg\\german"

    files = os.listdir(dir_path1)
    for file in files:
        print(file)

        file1 = os.path.join(dir_path1, file)
        file2 = os.path.join(dir_path2, file)

        f1 = open(file1, "r", encoding="utf-8")
        f2 = open(file2, "w", encoding="utf-8")

        lines = f1.readlines()
        f2.write(lines[0])
        for line in lines[1:]:
            for char in split_char:
                line = line.replace(char, "")
            f2.write(line)
        f1.close()
        f2.close()

    test("E:\\NLP\\code\\bilingual_word_emb\\data\\cross-lingual document classification"
         "\\rcv1\\rcv1\\19961218\\265961newsML.xml",
         "E:\\NLP\\code\\bilingual_word_emb\\data\\cross-lingual document classification\\rcv1\\test.txt")
    get_documents_done_with_label("E:\\NLP\\code\\bilingual_word_emb\\data\\cross-lingual document classification"
                                  "\\rcv2\\RCV2_Multilingual_Corpus\\japanese",
                                  "E:\\NLP\code\\bilingual_word_emb\\data\\cross-lingual document classification"
                                  "\\rcv2\\RCV2_Multilingual_Corpus_New\\japanese")
    languages = ["norwegian", "portuguese", "russian", "spanish", "spanish-latam", "swedish"]
    dir_path1 = "E:\\NLP\\code\\bilingual_word_emb\\data\\cross-lingual document classification" \
                "\\rcv2\\RCV2_Multilingual_Corpus\\"
    dir_path2 = "E:\\NLP\code\\bilingual_word_emb\\data\\cross-lingual document classification" \
                "\\rcv2\\RCV2_Multilingual_Corpus_New\\"
    for language in languages:
        get_documents_done_with_label(dir_path1 + language, dir_path2 + language)
    
    languages = ["chinese_simple", "danish", "dutch", "french", "german", "italian", "japanese", "norwegian",
                 "portuguese", "russian", "spanish", "spanish-latam", "swedish"]
    dir_path1 = "E:\\NLP\\code\\bilingual_word_emb\\data\\cross-lingual document classification\\rcv2" \
                "\\RCV2_Multilingual_Corpus_New\\"
    dir_path2 = "E:\\NLP\\code\\bilingual_word_emb\\data\\cross-lingual document classification\\rcv2" \
                "\\RCV2_Multilingual_Corpus_Single_label\\"
    for language in languages:
        child_dir_path1 = dir_path1 + language
        child_dir_path2 = dir_path2 + language
        get_single_label_english_file(child_dir_path1, child_dir_path2)
    
    get_single_label_english_file("E:\\NLP\code\\bilingual_word_emb\\data\\cross-lingual document classification"
                                  "\\rcv1\\rcv1_new",
                                  "E:\\NLP\\code\\bilingual_word_emb\\data\\cross-lingual document classification"
                                  "\\rcv1\\rcv1_single_label")
    sample_documents("E:\\NLP\\code\\bilingual_word_emb\\data\\cross-lingual document classification\\"
                     "rcv1\\rcv1_single_label",
                     "E:\\NLP\\code\\bilingual_word_emb\\data\\cross-lingual document classification\\"
                     "rcv1\\rcv1_sample", 34000)
    """
    sample_documents("E:\\NLP\\code\\bilingual_word_emb\\data\\cross-lingual document classification\\rcv2\\"
                     "RCV2_Multilingual_Corpus_Single_label\\french",
                     "E:\\NLP\\code\\bilingual_word_emb\\data\\cross-lingual document classification\\"
                     "rcv2\\RCV2_sample\\french", 42753)
    number = count_words_number("E:\\NLP\\code\\bilingual_word_emb\\data\\cross-lingual document classification\\"
                                "rcv2\\RCV2_sample\\french")
    print("There is total number of %d words in chinese\n" % number)
    """
    segment_words("E:\\NLP\\code\\bilingual_word_emb\\data\\cross-lingual document classification\\rcv1\\rcv1_sample",
                  "E:\\NLP\\code\\bilingual_word_emb\\data\\cross-lingual document classification\\"
                  "rcv1\\rcv1_sample_segments")
    """















