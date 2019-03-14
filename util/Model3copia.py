import pickle

import numpy as np
import pandas as pd
import ast
import gensim
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Embedding, LSTM, Input, Flatten, Dropout, TimeDistributed, Activation, Lambda, \
    Concatenate, add
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import LambdaCallback
import keras.backend as K
from keras.utils import to_categorical

from graphics import ModelGraphs

try:
    with open('./../word2vec.pickle', 'rb') as handle:
        word2vec = pickle.load(handle)
except FileNotFoundError:
    word2vec = gensim.models.KeyedVectors.load_word2vec_format('./../resources/SBW-vectors-300-min5.bin',
                                                               binary=True)
    with open('./../word2vec.pickle', 'wb') as handle:
        pickle.dump(word2vec, handle, protocol=pickle.HIGHEST_PROTOCOL)


def train_model_3():
    vector_dim = 300
    windows_size = 7
    data = pd.read_csv('./../resources/similarities.csv', sep='\t')

    data = data.sample(frac=1).reset_index(drop=True)
    data['t1'] = data['t1'].apply(ast.literal_eval)
    data['t2'] = data['t2'].apply(ast.literal_eval)
    data["input1"] = pad_sequences(data["t1"], value=0, padding="post", maxlen=windows_size).tolist()
    data["input2"] = pad_sequences(data["t2"], value=0, padding="post", maxlen=windows_size).tolist()
    data['category'] = round(data['entailing'] * 10)

    data['category'] = to_categorical(data['category'], 21).tolist()
    data['category'] = data['category'].apply(np.asarray)

    bins = [0, 100, 500, 700, 1000]
    data['category_r'] = pd.cut(data['relations'], bins, labels=[0, 1, 2, 3, 4])
    data['category_r'] = to_categorical(data['category_r'], 5).tolist()
    data['category_r'] = data['category_r'].apply(np.asarray)

    tot = len(data)
    val_num = round(tot * 15 / 100)
    test_num = round(tot * 10 / 100)
    train_num = tot - val_num - test_num

    train = data[0:train_num]
    val = data[train_num + 1: train_num + val_num]
    test = data[train_num + val_num + 1: tot]
    train_a = np.stack(train["input1"], axis=0)
    train_b = np.stack(train["input2"], axis=0)
    val_a = np.stack(val["input1"], axis=0)
    val_b = np.stack(val["input2"], axis=0)
    train_y = np.stack(train['category'], axis=0)
    train_y_r = np.stack(train['category_r'], axis=0)
    val_y = np.stack(val['category'], axis=0)
    val_y_r = np.stack(val['category_r'], axis=0)

    print(data.head())

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
    merged = add([lstm1, lstm2])
    dense1 = Dense(units=256, activation="relu")(merged)
    dense2 = Dense(units=256, activation="relu")(dense1)
    flatten1 = Flatten()(dense2)

    preds = Dense(21, activation='softmax')(flatten1)
    preds2 = Dense(5, activation='softmax')(flatten1)

    model = Model(inputs=[answer_inp, target_inp], outputs=[preds, preds2])

    # model.compile(optimizer="adam", loss='mean_squared_error', metrics=['mse', 'accuracy'])
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    bst_model_path = 'word2vec_model3_r.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path, verbose=True, save_best_only=True, save_weights_only=True)

    model_hist = model.fit([train_a, train_b], [train_y, train_y_r], verbose=True, validation_data=([val_a, val_b], [val_y, val_y_r]),
                           epochs=10, batch_size=156, shuffle=True,
                           callbacks=[early_stopping, model_checkpoint])

    ModelGraphs.plot_acc1(model_hist)

    model_json = model.to_json()
    with open("model3_r.json", "w") as json_file:
        json_file.write(model_json)

    return model_hist, model


def exponent_neg_manhattan_distance(left, right):
    return K.exp(-K.sum(K.abs(left - right), axis=1, keepdims=True))


def on_epoch_end(epoch, logs):
    print("\n\n\n")
