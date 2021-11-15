import pandas as pd

# example of a model defined with the sequential api
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Masking
from losses.weighted_cross_entropy import WeightedBinaryCrossEntropy


# Model inputs: sequence of n x m word embeddings
def create_model(lstm_layer_size=256, lstm_dropout=0.0, embedding_dim=100, output_dim=100, mask_value=0.0):
    """
    creates and returns a simple LSTM model
    :param embedding_dim: size of the input embeddings
    :return: model
    """
    model = Sequential()
    model.add(Masking(mask_value=mask_value, input_shape=(None, embedding_dim)))
    model.add(LSTM(lstm_layer_size, dropout=lstm_dropout, input_shape=(None, embedding_dim)))
    # model.add(Dense(256, activation="relu"))
    # model.add(Dense(128, activation="relu"))
    model.add(Dense(output_dim, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


if __name__ == "__main__":
    create_model()
    print("successfully created model")
