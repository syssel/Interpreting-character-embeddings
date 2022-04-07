import argparse
import logging
from collections import defaultdict

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

import matplotlib.pyplot as plt

import loaders as loader


def define_phonological_feature_sets(phonological_table, syllabic=False):
    is_vowel = np.argmax(phonological_table.filter(axis=1, items=['global_type_consonant', 'global_type_vowel']).to_numpy(), axis=1)
    consonants = []
    vowels = []
    for i, vowel_or_not in enumerate(is_vowel):
        if vowel_or_not == 1:
            vowels.append(phon.index[i])
        else:
            consonants.append(phon.index[i])

    ## Features that we want to consider
    feature_sets = {
    "global" : {"items":vowels+consonants, "features": {feature : list(filter(lambda x: x.startswith(feature), phonological_table.columns)) for feature in ("global_type",)}},
    "consonant" : {"items":consonants, "features": {feature : list(filter(lambda x: x.startswith(feature), phonological_table.columns)) for feature in ("consonant_voicing", "consonant_place", "consonant_manner")}},
    "vowel"     : {"items":vowels if not syllabic else consonants+vowels, "features": {feature : list(filter(lambda x: x.startswith(feature), phonological_table.columns)) for feature in ("vowel_height", "vowel_backness", "vowel_roundness")}}
    }

    return feature_sets


def leave_one_out(classifier, args, column, Y, X, model_dict):
    try:
        loo = LeaveOneOut()

        true_labels = []
        predictions = []
        targets = []

        for train_index, test_index in loo.split(X):
            x_train, y_train = X.iloc[train_index, :].to_numpy(), Y.iloc[train_index, :].to_numpy().flatten()
            x_test, y_test = X.iloc[test_index, :].to_numpy(), Y.iloc[test_index, :].to_numpy().flatten()

            target = list(X.iloc[test_index, :].index)

            clf = classifier(**args).fit(x_train, y_train)
            y_pred = clf.predict(x_test)

            predictions.append(y_pred)
            true_labels.append(y_test)
            targets.append(target)

        for z in zip(targets, true_labels, predictions):
            logging.info([i[0] for i in z])

        return predictions, true_labels

    except ValueError as e:
        logging.info("Could not fit model to"+column+":"+str(e))
        return None


def weight_visualizer(models):
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training classifiers to predict phonological features from character embeddings.')

    parser.add_argument( "-l", "--languages", nargs="*", type=str, default=["en"])
    parser.add_argument( "-e", "--embedding_types", nargs="*", type=str, default=["lstm"])
    parser.add_argument( "-b", "--baselines", nargs="*", type=str, default=["uniform", "most_frequent", "ipa"], choices=["uniform", "most_frequent"])

    args = parser.parse_args()

    loaders = {
        "ipa"  : loader.load_ipa,
        "shape": loader.load_shape,
        "ppmi" : loader.load_ppmi,
        "lstm" : loader.load_lstm,
        "transformer" : loader.load_transformer,
    }

    # Prepare output
    phon, _, _ =  loaders["ipa"]("en") # features are the same for all languages
    phonological_features = list(phon.columns)

    index_generator = lambda l, t, b="": "_".join(filter(None, [l,t,b]))
    indices = []
    for lang in args.languages:
        for embedding_type in args.embedding_types:
            indices.append(index_generator(lang, embedding_type))
            for baseline in args.baselines:
                indices.append(index_generator(lang, embedding_type, baseline))


    result_df = pd.DataFrame(columns=["Examples"]+indices, index=phonological_features)
    model_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None))) # lang:embedding:feature:model

    OUTDIR = "tables/"
    OUTDIR_viz = "plots/"
    RANDOM_STATE = 0
    outname = OUTDIR+"_".join(args.languages+args.embedding_types)
    open(outname+'.log', "w")
    logging.basicConfig(filename=outname+'.log',level=logging.INFO)

    for lang in args.languages:
        for embedding_type in args.embedding_types:
            rng = np.random.default_rng(RANDOM_STATE)

            print()
            print("Training {}:{}".format(lang, embedding_type))
            print("="*20)

            # Load embeddings and phonological features
            E, _, in_vocab = loaders[embedding_type](lang)
            phon, _, ipa_vocab =  loaders["ipa"](lang)

            vocabulary = sorted(list(set.intersection(set(ipa_vocab), set(in_vocab))))
            phon = phon.filter(axis=0, items=vocabulary)
            E = E.filter(axis=0, items=vocabulary)

            # Define consonants and vowels
            feature_set = define_phonological_feature_sets(phon, syllabic=lang=="ja")

            # Storing full models for visualisation and zero-shot
            full_models = defaultdict(lambda:None)
            full_models_baseline = defaultdict(lambda:defaultdict(lambda:None))

            # Training classifiers to predict feature
            for type, values in feature_set.items():

                # Filter input for character w.r.t. feature type (i.e., consonant or vowel)
                X = E.filter(axis=0, items=values["items"])

                for feature, columns in values["features"].items():
                    for column in columns:

                        # 1. TRAINING CLASSIFIERS (leave one out)
                        logging.info(":".join([lang, embedding_type, column]))
                        Y = phon.filter(axis=0, items=X.index).filter(axis=1, items=[column,])
                        result_df["Examples"][column] = np.sum(Y.to_numpy())

                        classifier = LogisticRegression
                        clf_args = {"random_state":RANDOM_STATE, "max_iter":1000}
                        output = leave_one_out(classifier, clf_args, column, Y, X, model_dict)

                        if output != None: predictions, true_labels = output
                        else: continue

                        ## Write results to table
                        result_df[index_generator(lang, embedding_type)][column] = f1_score(true_labels, predictions)

                        ## Training full model for visualizing weights and zero-shot
                        logging.info("Training full model...")
                        try:
                            full_models[column] = classifier(**clf_args).fit(X.to_numpy(), Y.to_numpy().flatten())
                        except ValueError as e:
                            logging.info("Could not fit model to"+column+":"+str(e))

                        ## Baselines...
                        for baseline_type in args.baselines:
                            logging.info(":".join([lang, embedding_type, baseline_type, column]))

                            if baseline_type == "uniform":
                                f1_scores_uniform= []
                                true_labels = Y.to_numpy().flatten()
                                for i in range(1000):
                                    predictions = rng.integers(0, high=1, size=Y.shape, endpoint=True)
                                    f1_scores_uniform.append(f1_score(true_labels, predictions))
                                result_df[index_generator(lang, embedding_type, baseline_type)][column] = np.mean(f1_scores_uniform)
                            else:

                                if baseline_type == "ipa":
                                    classifier = LogisticRegression
                                    clf_args = {"random_state":RANDOM_STATE, "max_iter":1000}
                                    output = leave_one_out(classifier, clf_args, column, Y, phon.filter(axis=0, items=X.index), model_dict)

                                elif baseline_type in ("most_frequent", "stratified"):
                                    classifier = DummyClassifier
                                    clf_args = {"strategy":baseline_type, "random_state":RANDOM_STATE}
                                    output = leave_one_out(classifier, clf_args, column, Y, X, model_dict)

                                    full_models_baseline[baseline_type][column] = classifier(**clf_args).fit(X.to_numpy(), Y.to_numpy().flatten())

                                if output != None: predictions, true_labels = output
                                else: continue

                                result_df[index_generator(lang, embedding_type, baseline_type)][column] = f1_score(true_labels, predictions)

            # 2. VISUALIZING CLASSIFIER WEIGHTS
            fig, axs = plt.subplots(nrows=len(full_models), ncols=1, figsize=(50,10), sharex=True, sharey=True)
            for i, (feature, model) in enumerate(full_models.items()):
                w = np.array(model.coef_).flatten()
                pcm = axs[i].pcolormesh(w.reshape((1, len(w))), vmin=-1, vmax=1)
                #axs[n].set(ylabel=column)
                axs[i].set_ylabel(feature, rotation=0, horizontalalignment="right")
                axs[i].set_yticks([])
            fig.savefig(OUTDIR_viz+"_".join([lang, embedding_type])+".png", dpi=300, bbox_inches="tight")


            # 3. ZERO-SHOT LEARNING between vowel and consonant features
            print()
            print("ZERO-SHOT EXPERIMENT")
            print()
            X_consonants = E.filter(axis=0, items=feature_set["consonant"]["items"])
            X_vowels = E.filter(axis=0, items=feature_set["vowel"]["items"])

            ## First, lets see how well the classifier for
            # predicting voicing for consonants is able to
            # predict whether a character is a vowel
            model1 = full_models["consonant_voicing_voiced"]

            print("char", "[+voiced]", sep="\t")
            predicted = []
            true_label = []
            for vowel, embedding in X_vowels.iterrows():
                pred = model1.predict([embedding.to_numpy()])[0]
                predicted.append(pred)
                true_label.append(1)
                print(vowel, pred, sep="\t") # model2.predict([embedding.to_numpy()])

            most_frequent = full_models_baseline["most_frequent"]["consonant_voicing_voiced"].predict(X_vowels)

            print()
            print("zero-shot:", f1_score(true_label, predicted), precision_score(true_label, predicted), recall_score(true_label, predicted))
            print("most frequent:", f1_score(true_label, most_frequent))
            print("random:", np.mean([f1_score(true_label, rng.integers(0, high=1, size=len(true_label), endpoint=True)) for _ in range(1000)]))
            print()
            print()
            ## Second, we will try to see how well the
            # classifier for predicting vowel roundness is
            # able to predict whether a consonant is labial
            # or not

            if lang == "en": continue # English does not have enough rounded vowels

            print("char", "[+round]", "is_labial", sep="\t")
            model1 = full_models["vowel_roundness_rounded"]
            #model2 = full_models["vowel_roundness_unrounded"]
            predicted = []
            true_label = []
            for consonant, embedding in X_consonants.iterrows():
                place_features = phon.filter(axis=0, items=consonant).filter(axis=1, like="consonant_place")
                labial_values = place_features.filter(axis=1, like="lab")
                is_labial = np.sum(labial_values.to_numpy())
                pred = model1.predict([embedding.to_numpy()])[0]
                prob = model1.predict_proba([embedding.to_numpy()])[0][-1]

                true_label.append(is_labial)
                predicted.append(pred)

                print(consonant, "{} ({:.2f})".format(pred, prob), "", is_labial, sep="\t") #model2.predict([embedding.to_numpy()])

            most_frequent = full_models_baseline["most_frequent"]["vowel_roundness_rounded"].predict(X_consonants)
            print()
            print("zero-shot:", f1_score(true_label, predicted), precision_score(true_label, predicted), recall_score(true_label, predicted))
            print("most frequent:", f1_score(true_label, most_frequent))
            print("random:", np.mean([f1_score(true_label, rng.integers(0, high=1, size=len(true_label), endpoint=True)) for _ in range(1000)]))
            print()
            print()


    df = result_df.dropna(axis=0, how='all')
    df_no_count_filtered = result_df.loc[:, df.columns != 'Examples'].dropna(axis=0, how='all')
    df_filtered = result_df.filter(axis=0, items=df_no_count_filtered.index)
    print()
    print("RESULT FROM PROBING TASK")
    print(df_filtered)
    latex_table = df_filtered.to_latex(float_format="{:0.2f}".format, bold_rows=True) #column_format='l'+"p{1.2cm}"*len(df.columns))

    f = open(outname+".txt", "w")
    f.write(latex_table)
    f.close()

    main_columns = ["global_type_consonant", "global_type_vowel", "consonant_voicing_voiced", "consonant_voicing_voiceless", "vowel_roundness_rounded", "vowel_roundness_unrounded"]
    all_columns = ["global_type_consonant", "global_type_vowel", "consonant_voicing_voiced", "consonant_voicing_voiceless",  "consonant_place_alveolar", "consonant_place_alveolo-palatal", "consonant_place_bilabial", "consonant_place_labio-dental", "consonant_place_palatal", "consonant_place_velar", "consonant_manner_approximant", "consonant_manner_nasal", "consonant_manner_non-sibilant-fricative", "consonant_manner_plosive", "consonant_manner_sibilant-fricative", "vowel_height_close", "vowel_height_close-mid", "vowel_height_mid", "vowel_height_open-mid", "vowel_backness_front", "vowel_backness_back", "vowel_roundness_rounded", "vowel_roundness_unrounded"]

    results = pd.DataFrame(data=df_filtered.T, columns=main_columns)
    latex_table = results.to_latex(float_format="{:0.2f}".format, bold_rows=True)

    f = open(outname+".tex", "w")
    f.write(latex_table)
    f.close()


    results = pd.DataFrame(data=df_filtered.T, columns=all_columns)
    latex_table = results.to_latex(float_format="{:0.2f}".format, bold_rows=True)

    f = open(outname+"_appendix.tex", "w")
    f.write(latex_table)
    f.close()
