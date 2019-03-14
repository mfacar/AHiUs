import ast
import pickle

import gensim
import pandas as pd

try:
    with open('./../word2vec.pickle', 'rb') as handle:
        word2vec = pickle.load(handle)
except FileNotFoundError:
    word2vec = gensim.models.KeyedVectors.load_word2vec_format('./../resources/SBW-vectors-300-min5.bin',
                                                               binary=True)
    with open('./../word2vec.pickle', 'wb') as handle:
        pickle.dump(word2vec, handle, protocol=pickle.HIGHEST_PROTOCOL)


def tokenize(row):
    word_len = len(row["words"])
    word_source = []
    word_2_vec = []
    for i in range(0, word_len):
        word = row["words"][i] if row["words"][i] in word2vec.vocab.keys() else row["lemmas"][0] if row["lemmas"][
                                                                                               0] in word2vec.vocab.keys() else "UNK"
        word_source.append(word)
        word_2_vec.append(word2vec.vocab[word].index)
    return word_source, word_2_vec


def embed_sentences(file_path):
    data = pd.read_csv(file_path, sep='\t')

    data.columns = ['index', 'description', 'cleaned', 'words', 'lemmas', 'pos', 'root']
    data['words'] = data['words'].apply(ast.literal_eval)
    data['lemmas'] = data['lemmas'].apply(ast.literal_eval)
    data['pos'] = data['pos'].apply(ast.literal_eval)

    new_columns = [tokenize(row[1]) for row in data.iterrows()]
    new_columns = pd.DataFrame(new_columns)
    new_columns.columns = ['source', 'vectors']
    result = pd.concat([data, new_columns], axis=1, sort=False)

    result.to_csv('../resources/embedding.csv', sep='\t')
