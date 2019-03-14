import numpy as np

import spacy
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences

from util.UserStoryChecker import text_analysis


def predict_simi():
    spacy_nlp = spacy.load('es')
    model_path = "./../resources/model.json"
    weights_path = "./../resources/word2vec_model.h5"

    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json)
    model.load_weights(weights_path)

    out = False

    while not out:
        sen1 = input('HU1: ')
        out = sen1 == "salir"
        sen2 = input('HU2: ')
        words1, lemmas1, pos1, root1, word_source1, word_2_vec1, correctness1, en_count1 = text_analysis(sen1, spacy_nlp)
        words2, lemmas2, pos2, root2, word_source2, word_2_vec2, correctness2, en_count2 = text_analysis(sen2, spacy_nlp)
        sequences_input1 = pad_sequences([word_2_vec1], value=0, padding="post", maxlen=16).tolist()
        sequences_input2 = pad_sequences([word_2_vec2], value=0, padding="post", maxlen=16).tolist()

        input1_a = np.asarray(sequences_input1)
        input2_a = np.asarray(sequences_input2)
        pred = model.predict([input1_a, input2_a], batch_size=None, verbose=0, steps=None)

        print(pred)

def predict_sim():
    spacy_nlp = spacy.load('es')
    model_path = "./../resources/model3.json"
    weights_path = "./../resources/word2vec_model3.h5"

    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json)
    model.load_weights(weights_path)

    out = False

    while not out:
        sen = input('HU: ')
        out = sen == "salir"
        words, lemmas, pos, root, word_source, word_2_vec, correctness, en_count = text_analysis(sen, spacy_nlp)
        sequences_input = pad_sequences([word_2_vec], value=0, padding="post", maxlen=16).tolist()
        input_a = np.asarray(sequences_input)
        pred = model.predict([input_a], batch_size=None, verbose=0, steps=None)

        results = {"plantilla": pred[0][0][0],
                   "valiosa": pred[1][0][0],
                   "independiente": pred[2][0][0],
                   "complejidad": pred[3][0][0],
                   "composicion": 1 - pred[4][0][0]}

        for key, value in results.items():
            print("{0}: {1:.0%}".format(key, value))


