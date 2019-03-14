import pickle
import gensim
import pandas as pd
import spacy
import hunspell

from util.SpellChecker import check_spell
from util.TextPreprocessor import clean_text
from util.UserStoryParser import parse_user_story

dic_es = hunspell.Hunspell("es_ES", "es_ES")
dic_en = hunspell.Hunspell("en_US", "en_US")

try:
    with open('./../word2vec.pickle', 'rb') as handle:
        word2vec = pickle.load(handle)
except FileNotFoundError:
    word2vec = gensim.models.KeyedVectors.load_word2vec_format('./../resources/SBW-vectors-300-min5.bin',
                                                               binary=True)
    with open('./../word2vec.pickle', 'wb') as handle:
        pickle.dump(word2vec, handle, protocol=pickle.HIGHEST_PROTOCOL)


def label_user_stories():
    user_stories_df = pd.read_csv("./../resources/data_aug.csv", sep="\t")

    user_stories_df["cleaned"] = user_stories_df.apply(lambda row: clean_text(row["description"]), axis=1)
    user_stories = [parse_user_story(us) for us in user_stories_df["cleaned"]]

    spacy_nlp = spacy.load('es')

    res = pd.DataFrame([text_analysis(text, spacy_nlp) for text in user_stories_df["cleaned"]],
                       columns=['words', 'lemmas', 'pos', 'root', 'word_source', 'word_2_vec', 'correctness', 'en_count'])

    valuable = [1 if us is not None and us.goal is not None else 0 for us in user_stories]
    structure = [1 if us is not None else 0 for us in user_stories]
    independent = [1 if us is not None and len(us.role.split()) <= 3 else 0 for us in user_stories]
    complex_goal = [1 if us is not None and len(us.role.split()) <= 10 else 0 for us in user_stories]
    feature = [us.feature if us is not None else None for us in user_stories]

    res_feature = pd.DataFrame([text_analysis(ft, spacy_nlp) for ft in feature],
                               columns=['words_feature', 'lemmas_feature', 'pos_feature', 'root_feature', 'word_source_feature', 'word_2_vec_feature', 'correctness_feature', 'en_count_feature'])

    conj_disj = [1 if " y " in us.feature or " o " in us.feature else 0 for us in user_stories]

    invest = pd.DataFrame(
        {'valuable': valuable, 'structure': structure, 'independent': independent,
         'complex_goal': complex_goal, 'feature': feature, 'conj_disj': conj_disj})

    result_df = pd.concat([res_feature, user_stories_df, res, invest], axis=1, sort=False)

    result_df.to_csv('../resources/data_total.csv', sep='\t')


def text_analysis(text, nlp):
    words = []
    lemmas = []
    pos = []
    root = ""

    text = clean_text(text)

    for token in nlp(text):
        words.append(token.text)
        lemmas.append(token.lemma_)
        pos.append(token.pos_)
        root = token.sent.root

    word_source, word_2_vec = tokenize(words, lemmas)

    try:
        correctness, en_count = check_spell(word_source)
    except:
        return words, lemmas, pos, root, word_source, word_2_vec, 0, 0

    return words, lemmas, pos, root, word_source, word_2_vec, correctness, en_count


def tokenize(words, lemmas):
    word_len = len(words)
    word_source = []
    word_2_vec = []

    for i in range(0, word_len):
        word = words[i] if words[i] in word2vec.vocab.keys() else lemmas[i] if lemmas[i] in word2vec.vocab.keys() else "NAN"
        word_source.append(word)
        word_2_vec.append(word2vec.vocab[word].index)

    return word_source, word_2_vec

