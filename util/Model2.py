import pickle

import numpy as np
import pandas as pd
import ast
import gensim
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Embedding, LSTM, Input, Flatten, Dropout, TimeDistributed, Activation, Lambda
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import LambdaCallback
import keras.backend as K

from graphics import ModelGraphs

try:
    with open('./../word2vec.pickle', 'rb') as handle:
        word2vec = pickle.load(handle)
except FileNotFoundError:
    word2vec = gensim.models.KeyedVectors.load_word2vec_format('./../resources/SBW-vectors-300-min5.bin',
                                                               binary=True)
    with open('./../word2vec.pickle', 'wb') as handle:
        pickle.dump(word2vec, handle, protocol=pickle.HIGHEST_PROTOCOL)


def train_model_2():
    vector_dim = 300
    windows_size = 10
    data = pd.read_csv('./../resources/feature_data.csv', sep='\t')

    data['word_2_vec_feature'] = data['word_2_vec_feature'].apply(ast.literal_eval)
    data["input"] = pad_sequences(data["word_2_vec_feature"], value=0, padding="post", maxlen=windows_size).tolist()
    data = data.sample(frac=1).reset_index(drop=True)
    train_a = np.stack(data["input"], axis=0)

    data['target'] = data['target'].apply(ast.literal_eval)
    data["target"] = pad_sequences(data["target"], value=0, padding="post", maxlen=windows_size).tolist()
    data = data.sample(frac=1).reset_index(drop=True)
    train_b = np.stack(data["target"], axis=0)

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

    target_inp = Input(shape=(windows_size,), dtype='int32')
    embedded_sequences_2 = embedding_layer(target_inp)
    bt2 = BatchNormalization()(embedded_sequences_2)

    lstm1 = lstm_layer(bt)
    lstm2 = lstm_layer(bt2)
    dense1 = Dense(units=256, activation="relu")(lstm1)
    dense2 = Dense(units=256, activation="relu")(lstm2)
    flatten1 = Flatten()(dense1)
    flatten2 = Flatten()(dense2)

    malstm_distance = Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
                             output_shape=lambda x: (x[0][0], 1))([flatten1, flatten2])

    model = Model(inputs=[answer_inp, target_inp], outputs=[malstm_distance])

    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    #    concats = concatenate([l1_out, l2_out], axis=-1)

    generation_callback = LambdaCallback(on_epoch_end=on_epoch_end)

    #early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    bst_model_path = 'word2vec_model2.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
    model_google_hist = model.fit([train_a, train_b], data["y"], verbose=True,
                                  epochs=100, batch_size=64, shuffle=True,
                                  callbacks=[model_checkpoint, generation_callback])

    model_json = model.to_json()
    with open("model2.json", "w") as json_file:
        json_file.write(model_json)

    return model_google_hist, model


def exponent_neg_manhattan_distance(left, right):
    return K.exp(-K.sum(K.abs(left - right), axis=1, keepdims=True))


def on_epoch_end(epoch, logs):
    print("\n\n\n")
