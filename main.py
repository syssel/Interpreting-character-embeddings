from distances import *
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import itertools
import argparse
from plots import *


parser=argparse.ArgumentParser()
parser.add_argument( "-e", "--embedding_types", nargs="*", type=str, default=["color", "lstm"])
parser.add_argument( "-l", "--languages", nargs="*", type=str, default=["en"])
args = parser.parse_args()

color_type = "average"

for lang in args.languages:
    print("LANGUAGE: ", lang)

    correlations = np.zeros(shape=(len(args.embedding_types),len(args.embedding_types)))
    pvalues = np.ones(shape=(len(args.embedding_types),len(args.embedding_types)))
    correlations_random_baseline = np.zeros(shape=(len(args.embedding_types),len(args.embedding_types)))
    pvalues_random_baseline = np.ones(shape=(len(args.embedding_types),len(args.embedding_types)))

    # load embeddings
    for embedding_combi in list(itertools.combinations(args.embedding_types, 2)):

        embeddings = {}
        print("\nCOMBINATION: ", embedding_combi)

        for idx, embedding_type in enumerate(embedding_combi):
            if embedding_type == "shape":
                distances, vocabulary = calculate_shape_distances(lang)
                embeddings["embeddings{0}".format(idx)] = [distances, vocabulary]
            elif embedding_type == "sound":
                distances, vocabulary = calculate_ipa_distances(lang)
                embeddings["embeddings{0}".format(idx)] = [distances, vocabulary]
            elif embedding_type == "color":
                distances, vocabulary = calculate_color_distances(lang, type=color_type)
                embeddings["embeddings{0}".format(idx)] = [distances, vocabulary]
            elif embedding_type == "ppmi":
                distances, vocabulary = calculate_ppmi_distances(lang)
                embeddings["embeddings{0}".format(idx)] = [distances, vocabulary]
            elif embedding_type == "lstm":
                distances, vocabulary = calculate_lstm_distances(lang)
                embeddings["embeddings{0}".format(idx)] = [distances, vocabulary]
            elif embedding_type == "bilstm":
                distances, vocabulary = calculate_bilstm_distances(lang)
                embeddings["embeddings{0}".format(idx)] = [distances, vocabulary]
            elif embedding_type == "transformer":
                distances, vocabulary = calculate_transformer_distances(lang)
                embeddings["embeddings{0}".format(idx)] = [distances, vocabulary]
            else:
                sys.exit("Embedding type unknown!")

        shared_vocabulary = np.intersect1d(embeddings["embeddings0"][1],embeddings["embeddings1"][1])
        print ("Shared vocabulary:",shared_vocabulary)
        print ("Shared vocabulary size:",len(shared_vocabulary))
        distances0 = embeddings["embeddings0"][0]
        distances1 = embeddings["embeddings1"][0]
        shared_distances = distances0.columns.intersection(distances1.columns)
        #print ("Shared distances:",shared_distances)
        print ("Shared distances size:",len(shared_distances))

        # calculate correlation
        print ("Pearson correlation, p-value:", pearsonr(distances0[shared_distances].iloc[0],distances1[shared_distances].iloc[0]))

        # calculate correlation from random distances
        print("RANDOM BASELINE")
        random_dists = np.random.rand(len(distances0[shared_distances].iloc[0]))
        print ("Pearson correlation, p-value:", pearsonr(random_dists,distances1[shared_distances].iloc[0]))

        correlations[args.embedding_types.index(embedding_combi[0]), args.embedding_types.index(embedding_combi[1])] = pearsonr(distances0[shared_distances].iloc[0],distances1[shared_distances].iloc[0])[0]
        pvalues[args.embedding_types.index(embedding_combi[0]), args.embedding_types.index(embedding_combi[1])] = pearsonr(distances0[shared_distances].iloc[0],distances1[shared_distances].iloc[0])[1]

        correlations_random_baseline[args.embedding_types.index(embedding_combi[0]), args.embedding_types.index(embedding_combi[1])] = pearsonr(random_dists,distances1[shared_distances].iloc[0])[0]
        pvalues_random_baseline[args.embedding_types.index(embedding_combi[0]), args.embedding_types.index(embedding_combi[1])] = pearsonr(random_dists,distances1[shared_distances].iloc[0])[1]

    plot_correlation_heatmap(correlations_random_baseline, args.embedding_types, "RANDOM", pvalues_random_baseline)
    plot_correlation_heatmap(correlations, args.embedding_types, lang, pvalues)
