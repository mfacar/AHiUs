import re
import spacy
import pandas as pd


def text_to_wordlist(stories_df):
    stories_df["cleaned"] = stories_df.apply(lambda row: clean_text(row[0]), axis=1)
    nlp = spacy.load('es')

    res = pd.DataFrame([tokenize(text, nlp) for text in stories_df["cleaned"]], columns=['words', 'lemmas', 'pos', 'root'])
    result = pd.concat([stories_df, res], axis=1, sort=False)

    result.to_csv('../resources/data.csv', sep='\t')
    return stories_df


def tokenize(text, nlp):
    words = []
    lemmas = []
    pos = []
    root = ""
    for token in nlp(text):
        words.append(token.text)
        lemmas.append(token.lemma_)
        pos.append(token.pos_)
        root = token.sent.root
    return words, lemmas, pos, root


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=áéíóúÁÉÍÓÚñ]", " ", text)
    text = re.sub(r" del ", " de el ", text)
    text = re.sub(r" al ", " a el ", text)
    text = re.sub(r" ej ", " ejemplo ", text)
    text = re.sub(r" el ", " ", text)
    text = re.sub(r" la ", " ", text)
    text = re.sub(r" los ", " ", text)
    text = re.sub(r" las ", " ", text)
    text = re.sub(r" un ", " ", text)
    text = re.sub(r" una ", " ", text)
    text = re.sub(r" unos ", " ", text)
    text = re.sub(r" unas ", " ", text)
    text = re.sub(r" de ", " ", text)
    text = re.sub(r" en ", " ", text)
    text = re.sub(r" que ", " ", text)
    text = re.sub(r" lo ", " ", text)
    text = re.sub(r" a ", " ", text)
    text = re.sub(r" se ", " ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"\<", " ", text)
    text = re.sub(r"\>", " ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text
