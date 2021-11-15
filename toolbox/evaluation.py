from typing import List

from sklearn.metrics import f1_score, hamming_loss, recall_score, precision_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import pickle


def binarize_model_output(y, threshold=0.5):
    return y > threshold

def evaluate_multiLabel(l_true, l_pred, printOutput):

    f1_micro = f1_score(l_true, l_pred, average='micro')
    f1_macro = f1_score(l_true, l_pred, average='macro')
    recall_micro = recall_score(l_true, l_pred, average='micro')
    recall_macro = recall_score(l_true, l_pred, average='macro')
    precision_micro = precision_score(l_true, l_pred, average='micro')
    precision_macro = precision_score(l_true, l_pred, average='macro')
    if printOutput:
        print(f" Macro Evaluation: f1_Score= {f1_macro} , Recall = {recall_macro} , Precision = {precision_macro}")
        print(f" Micro Evaluation: f1_Score= {f1_micro} , Recall = {recall_micro} , Precision = {precision_micro}")

    return f1_micro, precision_micro, recall_micro


def output_evaluation(model, sample_size, max_question_words, n_top_labels, l_true, predictions, normalize_embeddings,
                      learning_rate, vocab_size, n_epochs, thres=None, f1_pickle_name=None):
    """

    :param model:
    :param sample_size:
    :param max_question_words:
    :param n_top_labels:
    :param l_true:
    :param predictions:
    :param normalize_embeddings:
    :param learning_rate:
    :param vocab_size:
    :param n_epochs:
    :return:
    """

    print(f"Model Evaluation\n")
    print(f"normalize_embeddings = {normalize_embeddings}, learning_rate = {learning_rate}, vocab_size = {vocab_size}, epochs={n_epochs}")
    print(f"Parameter Settings:\n Sample size = {sample_size}, Max. number of words per question = {max_question_words}, Number of Top Labels used = {n_top_labels}\n")
    print(model.summary())
    if thres is None:
        max_f1, max_thres = optimize_thres(predictions, l_true, f1_pickle_name=f1_pickle_name)
        print(f"\nMetrics with optimized threshold of {max_thres}")
        l_pred = binarize_model_output(predictions, max_thres)
        evaluate_multiLabel(l_true, l_pred, True)
        return max_thres

    else:
        print(f"\nMetrics without optimized thres of {thres}")
        l_pred=binarize_model_output(predictions, thres)
        evaluate_multiLabel(l_true, l_pred, True)
        return thres


def optimize_thres(predictions, true_binary, plot=True, f1_pickle_name=None):
    max_f1 = -1
    max_thres = -1
    f1s = []
    precs = []
    recs = []
    thress = []
    for i in np.arange(0.0,1.0,0.01):
        pred_bin = binarize_model_output(predictions, i)
        f1, prec, rec = evaluate_multiLabel(true_binary, pred_bin, False)
        f1s.append(f1)
        precs.append(prec)
        recs.append(rec)
        thress.append(i)
        if f1 is not None and f1>max_f1:
            max_f1=f1
            max_thres=i

    if f1_pickle_name is not None:
        with open(f1_pickle_name, "wb") as out_file:
            pickle.dump((thress, f1s), out_file)

    if plot:
        plt.figure(figsize=(15, 8))
        plt.plot(thress, f1s, label="f1")
        plt.plot(thress, precs, label="precision")
        plt.plot(thress, recs, label="recall")
        plt.legend()
        plt.show()
    return max_f1, max_thres



def optimize_thres_reg(predictions, true_binary, plot=True, f1_pickle_name=None):
    max_f1 = -1
    max_thres = -1
    f1s = []
    precs = []
    recs = []
    accs = []
    thress = []
    for i in np.arange(0.0,1.0,0.01):
        pred_bin = binarize_model_output(predictions, i)
        f1 = f1_score(true_binary, pred_bin)
        acc = accuracy_score(true_binary, pred_bin)
        precs.append(precision_score(true_binary, pred_bin))
        recs.append(recall_score(true_binary, pred_bin))
        f1s.append(f1)
        accs.append(acc)
        thress.append(i)
        if f1 is not None and f1>max_f1:
            max_f1=f1
            max_thres=i

    if f1_pickle_name is not None:
        with open(f1_pickle_name, "wb") as out_file:
            pickle.dump((thress, f1s), out_file)

    if plot:
        plt.figure(figsize=(15, 8))
        plt.plot(thress, f1s, label="f1")
        plt.plot(thress, precs, label="precision")
        plt.plot(thress, recs, label="recall")
        plt.legend()
        plt.show()
    return max_f1, max_thres, max(accs)


def compare_f1_plots(f1_pickle_files: List[str], labels: List[str]):
    """
    Loads a number of f1 pickle files for evaluation and displays them along with the given matching tags
    :param f1_pickle_files: list of pickle files containing (thresholds, f1_scores) tuples
    :param labels: list of labels to attach to the f1 plots. Must be of same length as f1_pickle_files
    """
    thress = None
    all_f1s = []
    for f in f1_pickle_files:
        with open(f, "rb") as in_file:
            ts, f1s = pickle.load(in_file)
            if thress is None:
                thress = ts
            else:
                if not all([a == b for a, b in zip(thress, ts)]):
                    raise Exception("Different threshold between f1 pickles!")
            all_f1s.append(f1s)

    colors = [f"C{i}" for i in range(len(all_f1s))]
    plt.figure(figsize=(16, 8))
    for f1s, label, c in zip(all_f1s, labels, colors):
        plt.plot(thress, f1s, label=label, color=c)
    plt.ylim(0.5, 0.7)
    plt.legend()
    plt.show()
