import pandas as pd

# example of a model defined with the sequential api
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, LSTM, Masking, concatenate
from losses.weighted_cross_entropy import WeightedBinaryCrossEntropy

# Model inputs: sequence of n x m word embeddings


def create_model(lstm_layer_size=256, embedding_dim=100, output_dim=1, mask_value=0.0, num_mid_dense=0, lstm_dropout=0):
    """
    creates and returns a simple LSTM model
    :param embedding_dim: size of the input embeddings
    :return: model
    """
    x_arg = Input(shape=(None, embedding_dim))
    x_kp = Input(shape=(None, embedding_dim))
    
    mask_arg = Masking(mask_value=mask_value)(x_arg)
    mask_kp = Masking(mask_value=mask_value)(x_kp)
    
    lstm_arg = LSTM(lstm_layer_size, dropout=lstm_dropout)(mask_arg)
    lstm_kp = LSTM(lstm_layer_size, dropout=lstm_dropout)(mask_kp)
    
    concat = concatenate([lstm_arg, lstm_kp])

    next = concat
    for i in range(num_mid_dense):
        next = Dense(128, activation="relu")(next)
    
    output = Dense(output_dim, activation="sigmoid")(next)
    
    model = Model(inputs=[x_arg, x_kp], outputs=[output])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model
