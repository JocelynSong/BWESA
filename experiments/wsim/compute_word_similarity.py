import os

from experiments.wsim.ranking import *


def read_word_vectors(filename):
    vectors = {}
    f = open(filename, "r", encoding="utf-8")
    for line in f.readlines():
        words = line.strip().split()
        key = words[0].strip()
        embedding = []
        for word in words[1:]:
            embedding.append(float(word.strip()))
        vectors[key] = embedding
    return vectors


def main():
    word_sim_dir = "E:\\NLP\\code\\bilingual_word_emb\\data\\word similarity"
    file_lists = {"wordsim353.txt", "MC.txt", "RG.txt", "scws.txt", "rare.txt"}

    print("reading vectors\n")
    # dir_path = "E:\\NLP\\code\\bilingual_word_emb\\code\\sent_based_bwe\\data\\de-en\\embedding"

    vectors = read_word_vectors("E:\\NLP\\code\\bilingual_word_emb\\code\\sent_based_bwe\\data\\de-en\\embedding"
                                "\\en.emb.de_en_dimension128_alpha0.70_beta0.30_gama1.00_lambda4.00_epoch20.vec")

    print("compute correlation coefficient\n")
    for file in file_lists:
        print(file + ":")
        wsim_file_path = os.path.join(word_sim_dir, file)
        not_found = 0
        total_size = 1
        manual_dict, auto_dict = {}, {}
        f = open(wsim_file_path, "r", encoding="utf-8")
        for line in f.readlines():
            words = line.strip().split()
            word1 = words[0]
            word2 = words[1]
            value = float(words[2])
            if word1 in vectors and word2 in vectors:
                manual_dict[(word1, word2)] = value
                auto_dict[(word1, word2)] = cosine_sim(vectors[word1], vectors[word2])
            else:
                not_found += 1
            total_size += 1
        print("%20s" % file, "%15s" % str(total_size))
        print("%15s" % str(not_found))
        print("%15.4f" % spearmans_rho(assign_ranks(manual_dict), assign_ranks(auto_dict)))


if __name__ == "__main__":
    main()
