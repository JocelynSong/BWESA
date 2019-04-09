import numpy as np


class Embedding(object):
    def __init__(self, path, normalize=True):
        self.wi, self.m = self.read_file(path)
        if normalize:
            self.normalize()
        self.dim = self.m.shape[1]

    def normalize(self):
        norm = np.sqrt(np.sum(self.m * self.m, axis=1))
        self.m = self.m / norm[:, np.newaxis]

    @staticmethod
    def get_word_emb(line):
        lemmas = line.strip().split()
        word = lemmas[0].strip()
        embedding = [float(lemma.strip()) for lemma in lemmas[1:]]
        return word, embedding

    def read_file(self, path):
        f = open(path, "r", encoding="utf-8")
        word2id = dict()
        embeddings = list()
        for line in f.readlines():
            word, embedding = self.get_word_emb(line)
            word2id[word] = len(word2id)
            embeddings.append(embedding)
        embeddings = np.array(embeddings)
        return word2id, embeddings

    def represent(self, w):
        if w in self.wi:
            return self.m[self.wi[w]]
        else:
            return np.zeros(self.dim)
