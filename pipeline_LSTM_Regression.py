import pandas as pd
from nltk.tokenize import word_tokenize
# from toolbox.data_prep_helpers import *
# from toolbox.evaluation import *
# from toolbox.training import grid_search_es
import itertools

from models.title_body_lstm import create_model
from toolbox.data_preparation import get_data, create_FastText_embeddings, get_data_reg
from toolbox.evaluation import binarize_model_output, optimize_thres_reg
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from sklearn.metrics import f1_score, accuracy_score

from gensim.models import fasttext
from gensim.test.utils import datapath
from gensim.models.fasttext import load_facebook_vectors

import fasttext
import fasttext.util
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import datetime
import time

import numpy as np


TRAIN = False



data_full, data_test = get_data_reg(
    "C:/Users/dschr/Desktop/KPA/key-point-analysis-and-explanations-for-quantitative-text-analysis/KPA_2021_shared_task")


data = data_full[data_full["arg_tokens"].apply(len) <= 50]
data = data_full[data_full["key_point_tokens"].apply(len) <= 50]



print(data.shape)


if TRAIN:
    wv = create_FastText_embeddings(data, "arg_tokens")
    wv.fill_norms()
else:
    # wv = fasttext.load_facebook_model(datapath("C:/Users/dschr/PycharmProjects/key-point-analysis-and-explanations-for-quantitative-text-analysis/models/wiki.en.bin"))
    ft = fasttext.load_model("cc.en.300.bin")
    fasttext.util.reduce_model(ft, 100)

if TRAIN:
    X_arg = data["arg_tokens"].apply(lambda x: np.array([ft.get_word_vector(w) for w in x]))
    X_kp = data["key_point_tokens"].apply(lambda x: np.array([ft.get_word_vector(w) for w in x]))


    padding_element = np.array([0.0] * X_arg.iloc[0].shape[-1])
    X_arg = pad_sequences(X_arg, padding="post", dtype='float32', value=padding_element)
    X_kp = pad_sequences(X_kp, padding="post", dtype='float32', value=padding_element)
    print(X_arg.shape)
    print(X_kp.shape)

    y = data["label"]
    print(y.shape)


    best_params = {'lstm_layer_size': 256, 'lstm_dropout': 0.0, 'num_mid_dense': 1, 'output_dim': 1}


    # Train and test split with two datasets. Requires the zip to make the split for the question body and the question title
    X_train_z, X_test_z, y_train, y_test = train_test_split(list(zip(X_arg, X_kp)), y, test_size=0.2)
    X_train = list(zip(*X_train_z))
    X_test = list(zip(*X_test_z))
    #Convert to numpy arrays
    X_train=[np.array(X_train[0]), np.array(X_train[1])]
    X_test=[np.array(X_test[0]), np.array(X_test[1])]
    print(X_train[0].shape)


    model = create_model(**best_params)
    print(model.summary())

    model_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir="logs/fit/" + model_name

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, verbose=0),
        TensorBoard(log_dir=log_dir, histogram_freq=1),
        ModelCheckpoint(filepath=f"checkpoints/{model_name}", monitor="val_loss", restore_best_weights=True, verbose=0)
    ]

if TRAIN:
    model.fit(x=X_train, y=y_train, batch_size=32, epochs=20, validation_data=[X_test, y_test], callbacks=callbacks)
else:
    model = tf.keras.models.load_model('checkpoints/20211113-182346/')


# predictions = model.predict(X_test)
#
# maxF1, thres, acc = optimize_thres_reg(predictions, y_test)
#
# print(f"Max F1 Score {maxF1}, with threshold: {thres}")

data_t = data_test[data_test["arg_tokens"].apply(len) <= 50]
data_t = data_test[data_test["key_point_tokens"].apply(len) <= 50]

X_arg = data_t["arg_tokens"].apply(lambda x: np.array([ft.get_word_vector(w) for w in x]))
X_kp = data_t["key_point_tokens"].apply(lambda x: np.array([ft.get_word_vector(w) for w in x]))

padding_element = np.array([0.0] * X_arg.iloc[0].shape[-1])
X_arg = pad_sequences(X_arg, padding="post", dtype='float32', value=padding_element)
X_kp = pad_sequences(X_kp, padding="post", dtype='float32', value=padding_element)
print(X_arg.shape)
print(X_kp.shape)

y = data_t["label"]
print(y.shape)

X_test= [np.array(X_arg), np.array(X_kp)]

predictions = model.predict(X_test)

maxF1, thres, acc = optimize_thres_reg(predictions, y)

print(f"Max F1 Score {maxF1}, with threshold: {thres}, acc = {acc}")










