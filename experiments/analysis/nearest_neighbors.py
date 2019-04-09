import os
import sys

import tensorflow as tf
import numpy as np


def get_word_embedding(line):
    lemmas = line.strip().split()
    word = lemmas[0].strip()
    embedding = []
    for lemma in lemmas[1:]:
        embedding.append(float(lemma.strip()))
    return word, embedding


def read_word_embedding(filename):
    f = open(filename, "r", encoding="utf-8")
    embeddings = []
    word2id = {}
    for line in f.readlines():
        word, embedding = get_word_embedding(line)
        embeddings.append(embedding)
        word2id[word] = len(word2id)
    id2word = {index: word for word, index in word2id.items()}
    return word2id, id2word, embeddings


def compute_neighbor_mono(embeddings, valid_dataset, valid_size, valid_examples, id2word, session):
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm

    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
    sim = session.run(similarity)
    for i in range(valid_size):
        valid_word = id2word[valid_examples[i]]
        top_k = 15
        nearest = (-sim[i, :]).argsort()[1: top_k + 1]
        log_str = "Nearest to %s:" % valid_word
        for k in range(top_k):
            close_word = id2word[nearest[k]]
            log_str = "%s %s," % (log_str, close_word)
        print("%s\n" % log_str)


def compute_neighbor_bilingual(src_embeddings, tar_embeddings, valid_dataset, valid_size, valid_examples,
                               src_id2word, tar_id2word, session):
    norm_src = tf.sqrt(tf.reduce_sum(tf.square(src_embeddings), 1, keep_dims=True))
    normalized_src_embeddings = src_embeddings / norm_src
    norm_tar = tf.sqrt(tf.reduce_sum(tf.square(tar_embeddings), 1, keep_dims=True))
    normalized_tar_emebddings = tar_embeddings / norm_tar

    valid_embeddings = tf.nn.embedding_lookup(normalized_src_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_tar_emebddings, transpose_b=True)
    sim = session.run(similarity)
    for i in range(valid_size):
        valid_word = src_id2word[valid_examples[i]]
        top_k = 15
        nearest = (-sim[i, :]).argsort()[: top_k]
        log_str = "Nearest to %s: " % valid_word
        for k in range(top_k):
            close_word = tar_id2word[nearest[k]]
            log_str = "%s %s, " % (log_str, close_word)
        print("%s\n" % log_str)


def main(_):
    dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath("nearest_neighbors.py"))))
    emb_path = os.path.join(dir_path, "data\\de-en\\embedding")

    src_file = os.path.join(emb_path, "de.emb.de_en_dimension128_alpha0.60_beta0.40_gama1.00_lambda1.00_epoch19.vec")
    tar_file = os.path.join(emb_path, "en.emb.de_en_dimension128_alpha0.60_beta0.40_gama1.00_lambda1.00_epoch19.vec")

    src_word2id, src_id2word, src_embeddings = read_word_embedding(src_file)
    tar_word2id, tar_id2word, tar_embeddings = read_word_embedding(tar_file)
    src_embeddings = tf.constant(src_embeddings, dtype=tf.float32)
    tar_embeddings = tf.constant(tar_embeddings, dtype=tf.float32)

    valid_size = 20
    valid_window = 200
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    session = tf.Session()

    # source:German
    print("Compute German:\n")
    compute_neighbor_mono(src_embeddings, valid_dataset, valid_size, valid_examples, src_id2word, session)
    compute_neighbor_bilingual(src_embeddings, tar_embeddings, valid_dataset, valid_size, valid_examples, src_id2word,
                               tar_id2word, session)

    # source: English
    print("Compute English:\n")
    compute_neighbor_mono(tar_embeddings, valid_dataset, valid_size, valid_examples, tar_id2word, session)
    compute_neighbor_bilingual(tar_embeddings, src_embeddings, valid_dataset, valid_size, valid_examples, tar_id2word,
                               src_id2word, session)


if __name__ == "__main__":
    tf.app.run(main=main, argv=[sys.argv[0]])




