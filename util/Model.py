import pickle

import numpy as np
import pandas as pd
import ast
import gensim
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Embedding, LSTM, Input, Flatten, Dropout, TimeDistributed, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from sklearn.cluster import KMeans
from keras.callbacks import LambdaCallback

from graphics import ModelGraphs

try:
    with open('./../word2vec.pickle', 'rb') as handle:
        word2vec = pickle.load(handle)
except FileNotFoundError:
    word2vec = gensim.models.KeyedVectors.load_word2vec_format('./../resources/SBW-vectors-300-min5.bin',
                                                               binary=True)
    with open('./../word2vec.pickle', 'wb') as handle:
        pickle.dump(word2vec, handle, protocol=pickle.HIGHEST_PROTOCOL)


def train_model():
    vector_dim = 300
    windows_size = 16
    data = pd.read_csv('./../resources/train.csv', sep='\t')

    data['word_2_vec'] = data['word_2_vec'].apply(ast.literal_eval)

    data["input"] = pad_sequences(data["word_2_vec"], value=0, padding="post", maxlen=windows_size).tolist()

    data = data.sample(frac=1).reset_index(drop=True)

    train_a = np.stack(data["input"], axis=0)
    train_y_struncture = data["correctness"]
    train_y_struncture = data["structure"]
    train_y_valuable = data["valuable"]
    train_y_independent = data["independent"]
    train_y_complex = data["complex_goal"]
    train_y_conj_disj = data["conj_disj"]

    vocab_size = len(word2vec.index2word)
    embedding = word2vec.vectors

    num_lstm = np.random.randint(175, 275)
    rate_drop_lstm = 0.15 + np.random.rand() * 0.25

    embedding_layer = Embedding(vocab_size,
                                vector_dim,
                                weights=[embedding],
                                input_length=windows_size,
                                trainable=False)

    lstm_layer = LSTM(num_lstm, return_sequences=True, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

    answer_inp = Input(shape=(windows_size,), dtype='int32')
    embedded_sequences_1 = embedding_layer(answer_inp)

    bt = BatchNormalization()(embedded_sequences_1)
    lstm = lstm_layer(bt)
    dense1 = Dense(units=256, activation="relu")(lstm)
    dense2 = Dense(units=256, activation="relu")(dense1)
    flatten = Flatten()(dense2)
    out1 = Dense(1, activation='sigmoid')(flatten)
    out2 = Dense(1, activation='sigmoid')(flatten)
    out3 = Dense(1, activation='sigmoid')(flatten)
    out4 = Dense(1, activation='sigmoid')(flatten)
    out5 = Dense(1, activation='sigmoid')(flatten)

    model = Model(inputs=[answer_inp], outputs=[out1, out2, out3, out4, out5])

    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    bst_model_path = 'word2vec_model.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
    model_google_hist = model.fit([train_a], [train_y_struncture, train_y_valuable, train_y_independent,
                                              train_y_complex, train_y_conj_disj], verbose=True,
                                  validation_split=0.2,
                                  epochs=100, batch_size=64, shuffle=True,
                                  callbacks=[early_stopping, model_checkpoint])

    #    malstm_distance = Merge(mode=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
    #                            output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

    ModelGraphs.plot_acc3(model_google_hist, title="Plantilla", num=3)
    ModelGraphs.plot_acc3(model_google_hist, title="Valiosa", num=4)
    ModelGraphs.plot_acc3(model_google_hist, title="Independiente", num=5)
    ModelGraphs.plot_acc3(model_google_hist, title="Complejidad", num=6)
    ModelGraphs.plot_acc3(model_google_hist, title="Composicion", num=7)

    ModelGraphs.plot_loss3(model_google_hist, title="Plantilla", num=3)
    ModelGraphs.plot_loss3(model_google_hist, title="Valiosa", num=4)
    ModelGraphs.plot_loss3(model_google_hist, title="Independiente", num=5)
    ModelGraphs.plot_loss3(model_google_hist, title="Complejidad", num=6)
    ModelGraphs.plot_loss3(model_google_hist, title="Composicion", num=7)

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    return model_google_hist, model

#
# def exponent_neg_manhattan_distance(left, right):
#    ''' Helper function for the similarity estimate of the LSTMs outputs'''
#    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))

# def on_epoch_end(epoch, logs):
#    print("\n\n\n")

# Primero, seleccionamos una secuencia al azar para empezar a predecir
# a partir de ella


# generation_callback = LambdaCallback(on_epoch_end=on_epoch_end)
