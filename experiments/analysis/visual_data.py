# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 15:08:14 2016

@author: song

visual data
"""
import time
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn import manifold
print(__doc__)


def read_freq_words(filename):
    freq_words = []
    f = open(filename, "r", encoding="utf-8")
    for line in f.readlines():
        freq_words.append(line.strip())
    return freq_words


def read_freq_word_embedding(filename, freq_words):
    f = open(filename, "r", encoding="utf-8")
    embeddings = list()
    labels = list()
    for line in f.readlines():
        lemmas = line.strip().split()
        word = lemmas[0]
        if word not in freq_words:
            continue
        labels.append(word)
        embedding = []
        for lemma in lemmas[1:]:
            embedding.append(float(lemma.strip()))
        embeddings.append(embedding)
    return labels, embeddings


def readfile(zh_file, en_file):
    f1 = open(zh_file, 'r')
    f2 = open(en_file, 'r')
    X = []
    y = []
    for line in f1.readlines():
        items = line.strip().split()
        y.append(items[0])
        vec = []
        for i in range(1, len(items)):
            vec.append(float(items[i]))
        X.append(vec)
    print(len(X))
    for line in f2.readlines():
        items = line.strip().split()
        y.append(items[0])
        vec = []
        for i in range(1, len(items)):
            vec.append(float(items[i]))
        X.append(vec)
    f1.close()
    f2.close()
    X = np.array(X)
    return X, y
    

def get_new_words(filename):
    f = open(filename, 'r')
    y = []
    for line in f.readlines():
        l = line.strip()
        y.append(l)
    f.close()
    return y


def get_distance(x1, x2):
    x1 = np.array(x1)
    x2 = np.array(x2)
    dist = np.sum(np.square(x1-x2))
    print(dist)
    return dist


# ----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(good_words, y, X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # plt.figure(figsize=(100,100))
    # plt.scatter(X[:40,0],X[:40,1],c='r',marker='s',s=50)
    # plt.scatter(X[40:,0],X[40:,1],c='b',marker='s',s=50)
    '''
    plt.plot(X[:40,0],X[:40,1],'r^')
    plt.plot(X[40:,0],X[40:,1],'b*')
    '''
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # font = FontProperties(fname=r"C:\\WINDOWS\\Fonts\\simhei.ttf", size=20)
    font = FontProperties(fname=r"C:\\WINDOWS\\Fonts\\ARIALN.ttf", size=20)
    for i in range(39):
        plt.annotate('%s' % y[i], xy=(X[i][0], X[i][1]),
                     # textcoords='offset points',
                     ha='center', va='center', bbox=dict(boxstyle="round", fc="w", color='g'),
                     color='g', fontproperties=font)
    # font = FontProperties(fname=r"C:\\Windows\\Fonts\\times.ttf", size=15)
    font = FontProperties(fname=r"C:\\WINDOWS\\Fonts\\timesbd.ttf", size=20)
    for i in range(39, 79):
        '''
        plt.annotate('%s' %y[i],xy=(X[i][0],X[i][1]),
                     #textcoords='offset points',
                     ha='center',va='center',bbox=dict(boxstyle="round", fc="w",color='olive'),color='olive',fontproperties=font)
        '''
        
        if y[i] not in good_words:
            plt.annotate('%s' % y[i], xy=(X[i][0], X[i][1]),
                         # textcoords='offset points',
                         ha='center', va='center', bbox=dict(boxstyle="round", fc="w", color='olive'),
                         color='olive', fontproperties=font)
        else:
            plt.annotate('%s' % y[i], xy=(X[i][0], X[i][1]),
                         # textcoords='offset points',
                         ha='center', va='center', bbox=dict(boxstyle="round", fc="w", color='midnightblue'),
                         color='midnightblue', fontproperties=font)

    plt.xlim([-0.1, 1.0])
    plt.ylim([-0.1, 1.0])
    plt.show()


def main():
    dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath("nearest_neighbors.py"))))
    emb_path = os.path.join(dir_path, "data\\de-en\\embedding")
    freq_path = os.path.join(dir_path, "data\\de-en\\visualize")

    src_file = os.path.join(emb_path, "de.emb.de_en_dimension128_alpha0.60_beta0.40_gama1.00_lambda1.00_epoch19.vec")
    tar_file = os.path.join(emb_path, "en.emb.de_en_dimension128_alpha0.60_beta0.40_gama1.00_lambda1.00_epoch19.vec")
    src_freq_file = os.path.join(freq_path, "en.freq.de.txt")
    tar_freq_file = os.path.join(freq_path, "en.freq.words.txt")

    src_freq_words = read_freq_words(src_freq_file)
    tar_freq_words = read_freq_words(tar_freq_file)
    print(src_freq_words)
    print(tar_freq_words)

    src_words, src_freq_emb = read_freq_word_embedding(src_file, src_freq_words)
    tar_words, tar_freq_emb = read_freq_word_embedding(tar_file, tar_freq_words)
    print(len(src_words))
    print(len(tar_words))

    features = src_freq_emb
    labels = src_words

    for word, embedding in zip(tar_words, tar_freq_emb):
        labels.append(word)
        features.append(embedding)

    features = np.array(features)

    n_samples, n_features = features.shape
    n_neighbors = 30
    print("feature shape:%d, %d\n" % (n_samples, n_features))
    print("length of labels:%d\n" % len(labels))

    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    t0 = time.time()
    x_tsne = tsne.fit_transform(features)

    good_words = ['efforts', 'programme', 'cooperation', 'work', 'committee', "region",
                  'need', 'development', 'countries', 'people', 'relations', 'economic', 'activities',
                  'organizations', 'recommendations', 'security', 'important', 'implementation', 'situation', 'social',
                  'report', 'peace', "special", "conference", "action", "institutions", "provide",
                  "issues", "rights", "regional", "provisions", "must", "government"]

    print("Starting draw picture\n")
    plt.figure()
    plt.xticks([])
    plt.yticks([])
    # plt.title('Baseline Model',color='dimgrey')
    # plt.xlabel('BWE')
    plot_embedding(good_words, labels, x_tsne, "t-SNE embedding of the digits (time %.2fs)" % (time.time() - t0))


if __name__ == "__main__":
    main()


