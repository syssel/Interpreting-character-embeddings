import pandas as pd
import numpy as np
import itertools
from scipy.spatial import distance
from clustering import hierarchical_cluster, tsne_clustering
import sys
import regex

import loaders

def calculate_color_distances(language, type="average"):
    """Load synesthesia data and calculate Euclidean distance for all CIE*L*u*v color codes between all characters.
    Returns dataframe with distances for all subjects individually or averaged and a list of characters."""

    language_data, _, _ = loaders.load_color(language)

    distances_colors_all = pd.DataFrame()
    all_col_vecs = {}
    all_graphemes = []

    if language == "English" or language == "Dutch" or language == "Spanish":
        max_chars = 26
    elif language == "Korean":
        max_chars = 24
    elif language == "Japanese":
        max_chars = 46

    #print(language_data['Subject'].unique())
    for subject in language_data['Subject'].unique():
        subject_data = language_data.loc[language_data['Subject'] == subject]

        # only take data from subjects with ALL characters
        #if len(subject_data) == max_chars:
        idx = 0

        # todo: convert to dictionary to include sujects with incomplete data
        color_vector = subject_data[['L', 'u', 'v']].values.tolist()
        col_vecs = np.array([x for x in color_vector])
        #print(subject_data['Grapheme'].tolist())

        for c, vec in zip(subject_data['Grapheme'].tolist(), col_vecs):
            c = c.lower()
            if c not in all_col_vecs:
                all_col_vecs[c] = [vec]
            else:
                all_col_vecs[c].append(vec)

        #We calculate elementwise distances between the shared vocabulary of both embedding tables
        #Supposing that we have a vocabulary of "A,B,C and D", results are saved like this:
        #dst_AB, dst_AC, dst_AD, dst_BC, dst_BD, dst_CD
        distances_colors_subj = distance.pdist(col_vecs,metric="euclidean")

        # Generate column names
        graphemes = "".join(subject_data['Grapheme'].unique()).lower()
        combinations = list(itertools.combinations(graphemes, 2))
        combinations = ["".join(x) for x in combinations]

        dictionary = dict(zip(combinations, distances_colors_subj))
        distances_colors_subj_df = pd.DataFrame(data=dictionary, index=[idx])
        distances_colors_subj_df['subject'] = subject
        distances_colors_all = distances_colors_all.append(distances_colors_subj_df)

        idx+=1

    if type == "individual_subjs":
        return distances_colors_all, [g for g in graphemes]
    elif type == "average":
        for char, vecs in all_col_vecs.items():
            vecs_avg = np.mean(np.array(vecs), axis=0)
            all_col_vecs[char] = vecs_avg
        # average distance across all subjects
        mean_distances_df = pd.DataFrame(distances_colors_all.mean().to_dict(),index=[distances_colors_all.index.values[-1]])
        mean_distances_df['subject'] = "MEAN"
        #hierarchical_cluster(list(all_col_vecs.values()), list(all_col_vecs.keys()), language, "color", subject="MEAN")
        # todo: double check: all tsne color plot are exactly the same for any language?
        tsne_clustering(list(all_col_vecs.values()), list(all_col_vecs.keys()), language, "color", subject="MEAN")
        return mean_distances_df.drop('subject', axis=1), list(all_col_vecs.keys())


def calculate_shape_distances(language):
    """Loading the shape embeddings"""

    _, char_vecs, vocab = loaders.load_shape(language)
    shape_distances = distance.pdist(char_vecs,metric="cosine")

    shape_dists = pd.DataFrame()

    all_graphemes = "".join(vocab)
    all_combinations = list(itertools.combinations(all_graphemes, 2))
    all_combinations = ["".join(x) for x in all_combinations]

    # cluster shape embeddings
    #tsne_clustering(char_vecs, all_graphemes, language, "shape")

    shape_dists = pd.DataFrame(dict(zip(all_combinations, shape_distances)), index=[0])

    return shape_dists, vocab



def calculate_ipa_distances(language):
    """"""
    _, char_vecs, vocab = loaders.load_ipa(language)
    ipa_distances = distance.pdist(char_vecs,metric="cosine")
    all_graphemes = "".join(vocab)
    all_combinations = list(itertools.combinations(all_graphemes, 2))
    all_combinations = ["".join(x) for x in all_combinations]

    # cluster IPA embeddings
    #hierarchical_cluster(char_vecs, all_graphemes, language, "IPA")
    #tsne_clustering(char_vecs, all_graphemes, language, "IPA")

    ipa_dists = pd.DataFrame(dict(zip(all_combinations, ipa_distances)), index=[0])

    return ipa_dists, vocab


def calculate_ppmi_distances(language):
    """Load file with PPMI character embeddings and calculate cosine distance between all characters.
    Returns dataframe with distances and a list of characters."""
    _, ppmi_vecs, vocab = loaders.load_ppmi(language)
    ppmi_distances = distance.pdist(ppmi_vecs, metric="cosine")

    all_graphemes = "".join(vocab)
    all_combinations = list(itertools.combinations(all_graphemes, 2))
    all_combinations = ["".join(x) for x in all_combinations]

    # cluster IPA embeddings
    #hierarchical_cluster(ppmi_vecs, all_graphemes, language, "IPA")
    tsne_clustering(ppmi_vecs, all_graphemes, language, "PPMI")

    ppmi_dists = pd.DataFrame(dict(zip(all_combinations, ppmi_distances)), index=[0])

    return ppmi_dists, vocab


def calculate_lstm_distances(language):
    """Load LSTM character embeddings and calculate cosine distance between all characters.
    Returns dataframe with distances and a list of characters."""

    _, char_vecs, vocab = loaders.load_lstm(language)
    lstm_distances = distance.pdist(char_vecs,metric="cosine")

    # Generate column names
    all_graphemes = "".join(vocab)
    all_combinations = list(itertools.combinations(all_graphemes, 2))
    all_combinations = ["".join(x) for x in all_combinations]

    # cluster LSTM embeddings
    #hierarchical_cluster(char_vecs, all_graphemes, language, "LSTM")
    tsne_clustering(char_vecs, all_graphemes, language, "LSTM")

    lstm_dists = pd.DataFrame(dict(zip(all_combinations, lstm_distances)), index=[0])

    return lstm_dists, vocab

def calculate_bilstm_distances(language):
    """Load LSTM character embeddings and calculate cosine distance between all characters.
    Returns dataframe with distances and a list of characters."""

    _, char_vecs, vocab = loaders.load_bilstm(language)
    lstm_distances = distance.pdist(char_vecs,metric="cosine")

    # Generate column names
    all_graphemes = "".join(vocab)
    all_combinations = list(itertools.combinations(all_graphemes, 2))
    all_combinations = ["".join(x) for x in all_combinations]

    # cluster LSTM embeddings
    #hierarchical_cluster(char_vecs, all_graphemes, language, "LSTM")
    tsne_clustering(char_vecs, all_graphemes, language, "biLSTM")

    lstm_dists = pd.DataFrame(dict(zip(all_combinations, lstm_distances)), index=[0])

    return lstm_dists, vocab

def calculate_transformer_distances(language):
    """Load file with transformer character embeddings and calculate cosine distance between all characters.
    Returns dataframe with distances and a list of characters."""

    _, char_vecs, vocab = loaders.load_transformer(language)
    trans_distances = distance.pdist(char_vecs,metric="cosine")

    # Generate column names
    all_graphemes = "".join(vocab)
    all_combinations = list(itertools.combinations(all_graphemes, 2))
    all_combinations = ["".join(x) for x in all_combinations]

    # cluster LSTM embeddings
    #hierarchical_cluster(char_vecs, all_graphemes, language, "LSTM")
    #tsne_clustering(char_vecs, all_graphemes, language, "Transformer")

    trans_dists = pd.DataFrame(dict(zip(all_combinations, trans_distances)), index=[0])

    return trans_dists, vocab
