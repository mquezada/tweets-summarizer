import gensim
import settings
import pathlib
import logging
import numpy as np
from collections import defaultdict


vectors = []
model = None


class Word2VecMock:
    def __init__(self):
        self.syn0norm = []
        self.index2word = defaultdict(str)

    def most_similar(self, word, topn=3):
        res = []
        for i in range(topn):
            res.append((word + "_%d" % i, 0.85))
        return res

    def __contains__(self, item):
        return True


def get_model():
    global model

    model_path = pathlib.Path(settings.W2V_PATH)
    if model_path.exists():
        logging.info("Loading word2vec from %s" % settings.W2V_PATH)
        model = gensim.models.Word2Vec.load(model_path)
        model.init_sims()
    else:
        logging.info("Loading word2vec mock")
        model = Word2VecMock()

    init_vectors()
    return model


def word2vec(word):
    if word in model:
        idx = model.vocab[word].index
        return vectors[idx]


def most_similar(word, topn=3):
    v_word = word2vec(word)
    sims = vectors.dot(v_word)
    indices = np.argpartition(sims, len(vectors) - topn)
    return [(model.index2word[i], sims[i]) for i in indices]


def init_vectors():
    if model:
        global vectors
        vectors = model.syn0norm
