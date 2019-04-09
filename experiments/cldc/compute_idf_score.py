import os


def count_idf(dir_path):
    word_idf = {}
    files = os.listdir(dir_path)
    total_number = len(files)
    print("There is total number of %d English files\n" % total_number)
    index = 0
    for file in files:
        index += 1
        if index % 1000 == 0:
            print("Now deal %d files\n" % index)
        filename = os.path.join(dir_path, file)
        f = open(filename, "r", encoding="utf-8")
        lines = f.readlines()[1:]
        words = []
        for line in lines:
            words.extend(line.strip().split())
        words = set(words)
        for word in words:
            if word in word_idf:
                word_idf[word] += 1
            else:
                word_idf[word] = 1
    for key, count in word_idf.items():
        word_idf[key] = count / float(total_number)
    return word_idf


def write_words_idf(word_idf, filename):
    f = open(filename, "w", encoding="utf-8")
    for key, score in word_idf.items():
        f.write(key + "|||" + str(score))
        f.write("\n")
    f.close()


def get_word_frequency(filename1, filename2):
    f = open(filename1, "r", encoding="utf-8")
    word_dict = {}
    ind = 0
    for line in f.readlines():
        ind += 1
        if ind % 10000 == 0:
            print(ind)
        words = line.strip().split()
        for word in words:
            if word in word_dict:
                word_dict[word] += 1
            else:
                word_dict[word] = 1
    word_dict = sorted(word_dict.items(), key=lambda d: d[1], reverse=True)
    f = open(filename2, "w", encoding="utf-8")
    for word, count in word_dict:
        if count > 2000:
            f.write(word + "|||" + str(count) + "\n")
    f.close()


if __name__ == "__main__":
    """
    words_idf = count_idf("E:\\NLP\\code\\bilingual_word_emb\\data\\cross-lingual document classification\\"
                          "rcv2\\RCV2_sample_seg\\chinese")
    write_words_idf(words_idf, "E:\\NLP\\code\\bilingual_word_emb\\experiment\\cldc\\data\\chinese.idf.scores")
    """
    get_word_frequency("E:\\NLP\\code\\bilingual_word_emb\\code\\sent_based_bwe\\data\\zh.raw",
                       "E:\\NLP\\code\\bilingual_word_emb\\code\\sent_based_bwe\\data\\freq\\zh.fre.3000")
    get_word_frequency("E:\\NLP\\code\\bilingual_word_emb\\code\\sent_based_bwe\\data\\en.raw",
                       "E:\\NLP\\code\\bilingual_word_emb\\code\\sent_based_bwe\\data\\freq\\en.fre.3000")
