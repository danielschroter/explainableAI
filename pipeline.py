import pandas as pd
from nltk.tokenize import word_tokenize
# from toolbox.data_prep_helpers import *
# from toolbox.evaluation import *
# from toolbox.training import grid_search_es
import itertools

from models.lstm_classifier import create_model
from toolbox.data_preparation import get_data, create_FastText_embeddings
from toolbox.evaluation import binarize_model_output, optimize_thres
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from gensim.models import fasttext
from gensim.test.utils import datapath
from gensim.models.fasttext import load_facebook_vectors

import fasttext
import fasttext.util

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import datetime
import time

import numpy as np


TRAIN = False

data_full = get_data(
    "C:/Users/dschr/Desktop/KPA/key-point-analysis-and-explanations-for-quantitative-text-analysis/KPA_2021_shared_task/kpm_data")

print(data_full.shape)
data = data_full[data_full["labels"].apply(lambda labels: all([isinstance(l, str) for l in labels]))]
print(data.shape)

data = data_full[data_full["arg_tokens"].apply(len) <= 50]


print(data.shape)


if TRAIN:
    wv = create_FastText_embeddings(data, "arg_tokens")
    wv.fill_norms()
else:
    # wv = fasttext.load_facebook_model(datapath("C:/Users/dschr/PycharmProjects/key-point-analysis-and-explanations-for-quantitative-text-analysis/models/wiki.en.bin"))
    ft = fasttext.load_model("cc.en.300.bin")
    fasttext.util.reduce_model(ft, 100)


X = data["arg_tokens"].apply(lambda x: np.array([ft.get_word_vector(w) for w in x]))

padding_element = np.array([0.0] * X.iloc[0].shape[-1])
X = pad_sequences(X, padding="post", dtype='float32', value=padding_element)
print(X.shape)

label_encoder = MultiLabelBinarizer()
label_encoder.fit(data["labels"])
y = label_encoder.transform(data["labels"])
print(y.shape)

model = create_model(embedding_dim=100, output_dim=207, mask_value=0.)
print(model.summary())

model_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir="logs/fit/" + model_name

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, verbose=0),
    TensorBoard(log_dir=log_dir, histogram_freq=1),
    ModelCheckpoint(filepath=f"checkpoints/{model_name}", monitor="val_loss", restore_best_weights=True, verbose=0)
]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(x=X_train, y=y_train, batch_size=128, epochs=50, validation_data=[X_test, y_test], callbacks=callbacks)


predictions = model.predict(X_test)

maxF1, thres = optimize_thres(predictions, y_test)


preds = model.predict(X_test[:20])

l_pred = label_encoder.inverse_transform(binarize_model_output(preds, threshold=thres))
l_true = label_encoder.inverse_transform(y_test[:20])
texts = data["arg_tokens"][:20]
raw_texts = data["argument"][:20]

for pred, act, txt, raw_txt in zip(l_pred, l_true, texts, raw_texts):
    print(f"TRUE: {act}\nPREDICTION: {pred}\n")
    print(txt)
    print(raw_txt)



print(f"Max F1 Score {maxF1}, with threshold: {thres}")

