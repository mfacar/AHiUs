import pickle
import spacy

import gensim
import pandas as pd
import ast

from util.DataTagger import clean_text
from util.TextPreprocessor import tokenize

nlp = spacy.load('es')

try:
    with open('./../word2vec.pickle', 'rb') as handle:
        word2vec = pickle.load(handle)
except FileNotFoundError:
    word2vec = gensim.models.KeyedVectors.load_word2vec_format('./../resources/SBW-vectors-300-min5.bin',
                                                               binary=True)
    with open('./../word2vec.pickle', 'wb') as handle:
        pickle.dump(word2vec, handle, protocol=pickle.HIGHEST_PROTOCOL)


def similar(row):
    source_words = row[0]
    words = row[1]
    sims = [(word, get_similar(word)) for word in words]
    clone_words = [clone(source_words[i], sims[i]) for i in range(0, len(words))]
    return clean_text(" ".join(clone_words))


def get_similar(word):
    try:
        return word2vec.similar_by_word(word)
    except KeyError:
        return [("", 1)]


def clone(word, sim):
    us_words = ["como","deseo","quiero","necesito","requiero","quisiera","necesita","para"]
    distance = sim[1][0][1]
    closest = sim[1][0][0]
    return word if word in us_words else closest if distance > 0.7 else word


def duplicate_data():
    stories = pd.read_csv("./../resources/UserStoriesDB.csv", encoding="utf-16", sep="\t")
    stories = stories.apply(lambda row: clean_text(row["description"]), axis=1)
    tokenized = stories.apply(lambda text: tokenize(text, nlp))
    result = pd.Series([similar(row) for row in tokenized])
    stories = pd.DataFrame(stories.append(result))
    stories.to_csv('../resources/data_aug.csv')
