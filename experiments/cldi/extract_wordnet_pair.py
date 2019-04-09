def read_wordnet(filename):
    f = open(filename, "r", encoding="utf-8")
    lines = f.readlines()[1:]
    f.close()
    word_net = {}
    for line in lines:
        print(line)
        symbols = line.strip().split("\t")
        if len(symbols) < 3:
            continue
        index = symbols[0].strip()
        word = symbols[2].strip()
        word_net[index] = word
    return word_net


def get_word_pair(en_name, zh_name):
    en_wordnet = read_wordnet(en_name)
    zh_wordnet = read_wordnet(zh_name)
    gold_pair_ = []
    for key, word in en_wordnet.items():
        if key in zh_wordnet.keys():
            gold_pair_.append([word, zh_wordnet[key]])
    return gold_pair_


def save_gold_pair(gold_pair_, save_file):
    f = open(save_file, "w", encoding="utf-8")
    for pair in gold_pair_:
        en_word, zh_word = pair
        f.write(en_word + "|||" + zh_word + "\n")
    f.close()


def change_stop_words_file_form(file1, file2):
    f1 = open(file1, "r", encoding="utf-8")
    line = f1.readline()
    words = line.strip().split(",")
    stop_words = [word.strip() for word in words]
    f2 = open(file2, "w", encoding="utf-8")
    for word in stop_words:
        f2.write(word + "\n")
    f2.close()


if __name__ == "__main__":
    gold_pair = get_word_pair("E:\\NLP\\code\\bilingual_word_emb\\data\\cross-lingual dictionary induction\\all"
                              "\\wns\\wikt\\wn-wikt-eng.tab",
                              "E:\\NLP\\code\\bilingual_word_emb\\data\\cross-lingual dictionary induction\\all"
                              "\\wns\\wikt\\wn-wikt-fra.tab")
    save_gold_pair(gold_pair, "E:\\NLP\\code\\bilingual_word_emb\\data\\cross-lingual dictionary induction\\gold_pair"
                              "\\en-fr.gold.pair")
    """
    change_stop_words_file_form("E:\\NLP\\code\\bilingual_word_emb\\data\\cross-lingual dictionary induction"
                                "\\stop-word-list-baidu-zh.txt",
                                "E:\\NLP\\code\\bilingual_word_emb\\data\\cross-lingual dictionary induction"
                                "\\stop_word_zh.txt")
    """

