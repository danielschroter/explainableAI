import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec, FastText
import pickle



def sent_tokenize_text(txt):
    """
    tokenize a single body of text that may consist of multiple sentences. This function will usually be applied to
    single question/answer bodies as data-preparation for training/applying embeddings
    :param txt: string containing some text
    :return: list of preprocessed single sentences
    """
    txt = ' '.join(txt.split())  # replaces all whitespace with single space (including line-breaks)
    sents = sent_tokenize(txt)
    sents = [s[:-1] if s.endswith(".") else s for s in sents]
    sents = [s.replace(",", "") for s in sents]
    return sents


def word_tokenize_sent(s):
    """tokenizes all words in a sentence (after lower-casing them)"""
    return [w.lower() for w in word_tokenize(s)]

def generate_question_level_tokens(txt):
    """
    Method generates a list where each entry is a list containing the words per question
    :param txt: txt to be tokenized
    :return: a flattened nested list containing the tokens per question
    """
    sents = sent_tokenize_text(txt)
    words = [word_tokenize_sent(s) for s in sents]
    return [item for sublist in words for item in sublist]  # flatten nested list



def get_data(path_to_dir):

    df_arguments = pd.read_csv(f"{path_to_dir}/arguments_train.csv")
    df_keypoints = pd.read_csv(f"{path_to_dir}./key_points_train.csv")
    df_labels = pd.read_csv(f"{path_to_dir}/labels_train.csv")

    # merged = pd.merge(df_arguments, df_labels, on="arg_id")
    merged = df_labels[df_labels["label"]==1]

    grouped_labels = merged.groupby("arg_id").apply(lambda group: group["key_point_id"].tolist())

    grouped_labels_df = pd.DataFrame({"arg_id": grouped_labels.index, "labels": grouped_labels.values})

    merged = pd.merge(grouped_labels_df, df_arguments, on="arg_id")
    merged["argument"] = merged["argument"].apply(lambda x: x.replace("\"", ""))
    merged["argument"] = merged["argument"].apply(lambda x: x.replace("\'", ""))



    merged["arg_tokens"] = merged["argument"].apply(generate_question_level_tokens)

    return merged


def get_data_reg(path_to_dir):

    df_arguments = pd.read_csv(f"{path_to_dir}/kpm_data/arguments_train.csv")
    df_keypoints = pd.read_csv(f"{path_to_dir}/kpm_data/key_points_train.csv")
    df_labels = pd.read_csv(f"{path_to_dir}/kpm_data/labels_train.csv")


    df_arguments_test = pd.read_csv(f"{path_to_dir}/test_data/arguments_test.csv")
    df_keypoints_test = pd.read_csv(f"{path_to_dir}/test_data/key_points_test.csv")
    df_labels_test = pd.read_csv(f"{path_to_dir}/test_data/labels_test.csv")

    # merged = pd.merge(df_arguments, df_labels, on="arg_id")

    # grouped_labels = df_labels.groupby("arg_id").apply(lambda group: group["key_point_id"].tolist())

    # grouped_labels_df = pd.DataFrame({"arg_id": grouped_labels.index, "labels": grouped_labels.values})

    merged = pd.merge(df_labels, df_arguments, on="arg_id")
    merged = pd.merge(merged, df_keypoints, on="key_point_id")
    merged["argument"] = merged["argument"].apply(lambda x: x.replace("\"", ""))
    merged["argument"] = merged["argument"].apply(lambda x: x.replace("\'", ""))
    merged["key_point"] = merged["key_point"].apply(lambda x: x.replace("\"", ""))
    merged["key_point"] = merged["key_point"].apply(lambda x: x.replace("\'", ""))



    merged["arg_tokens"] = merged["argument"].apply(generate_question_level_tokens)
    merged["key_point_tokens"] = merged["key_point"].apply(generate_question_level_tokens)


    merged_test = pd.merge(df_labels_test, df_arguments_test, on="arg_id")
    merged_test = pd.merge(merged_test, df_keypoints_test, on="key_point_id")
    merged_test["argument"] = merged_test["argument"].apply(lambda x: x.replace("\"", ""))
    merged_test["argument"] = merged_test["argument"].apply(lambda x: x.replace("\'", ""))
    merged_test["key_point"] = merged_test["key_point"].apply(lambda x: x.replace("\"", ""))
    merged_test["key_point"] = merged_test["key_point"].apply(lambda x: x.replace("\'", ""))



    merged_test["arg_tokens"] = merged_test["argument"].apply(generate_question_level_tokens)
    merged_test["key_point_tokens"] = merged_test["key_point"].apply(generate_question_level_tokens)


    return merged, merged_test



def create_FastText_embeddings(dataframe, textcolumn):
    #data = tokenize_text(dataframe, textcolumn)
    data = dataframe[textcolumn].tolist()
    model = FastText(min_count=1, vector_size=100, window=3, sg=1)
    model.build_vocab(corpus_iterable=data)
    model.train(corpus_iterable= data, total_examples=len(data), epochs=10)
    return model.wv



def load_fasttext_embeddings(path):
    """
    Loads trained embeddings
    :param path: path to the location of the embeddings
    :return:
    """
    with open(path, "rb") as in_file:
        return pickle.load(in_file)







