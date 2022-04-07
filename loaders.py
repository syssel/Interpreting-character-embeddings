import sys
import regex

import numpy as np
import pandas as pd

def load_shape(language):
    shape_char_embeddings = pd.read_csv("shape_embeddings/shape_embs-onefont-"+language+".txt", sep=',', header=0, index_col=0)
    #shape_char_embeddings = pd.read_csv("shape_embeddings/shape_embs-onefont-"+language+"-sylls.txt", sep=',', header=0, index_col=0)
    char_vecs = shape_char_embeddings.to_numpy()
    vocab = shape_char_embeddings.index
    return shape_char_embeddings, char_vecs, vocab

def load_ipa(language):
    lang_keys = {
        "en":"eng",
        "ja":"jpn",
        "es":"spa",
        "nl":"dut",
        "ko":"kor",
        "ko-syll":"kor_syllables",
    }
    ipa_char_embeddings = pd.read_csv("ipa_embeddings/"+lang_keys[language]+"_most_frequent.txt", sep=',', header=0, index_col="CHAR")
    char_vecs = ipa_char_embeddings.to_numpy()
    vocab = ipa_char_embeddings.index
    return ipa_char_embeddings, char_vecs, vocab

def load_ppmi(language):
    ppmi_char_embeddings = pd.read_csv("ppmi_embeddings/"+language+".txt", sep=',', header=0, index_col="CHAR")
    ppmi_vecs = ppmi_char_embeddings.to_numpy()
    vocab = ppmi_char_embeddings.index
    return ppmi_char_embeddings, ppmi_vecs, [str(v) for v in vocab]

def load_lstm(language):
    lstm_char_embeddings = pd.read_csv("lstm_embeddings/lstm_char_embeddings-"+language+"_3000000_100ep_2lay.txt", sep=',', quotechar='"', index_col="CHAR")
    #lstm_char_embeddings = pd.read_csv("lstm_embeddings/lstm_char_embeddings-"+language+"-noJamos_3000000_100ep.txt", sep=',', quotechar='"', index_col="CHAR")
    #print(lstm_char_embeddings.head())
    if language == "ja":
        for index, row in lstm_char_embeddings.iterrows():
            if regex.search(r'\p{IsHan}|\p{IsKatakana}', index):
                lstm_char_embeddings.drop(index, inplace=True)
    char_vector = lstm_char_embeddings.to_numpy()
    char_vecs = np.array([x for x in char_vector])
    vocab = lstm_char_embeddings.index
    return lstm_char_embeddings, char_vecs, vocab

def load_bilstm(language):

    lstm_char_embeddings = pd.read_csv("bilstm_embeddings/bilstm_char_embeddings-"+language+"_3000000_100ep_1lay.txt", sep=',', quotechar='"', index_col="CHAR")
    #print(lstm_char_embeddings.head())
    if language == "ja":
        for index, row in lstm_char_embeddings.iterrows():
            if regex.search(r'\p{IsHan}|\p{IsKatakana}', index):
                lstm_char_embeddings.drop(index, inplace=True)
    char_vector = lstm_char_embeddings.to_numpy()
    char_vecs = np.array([x for x in char_vector])
    vocab = lstm_char_embeddings.index
    return lstm_char_embeddings, char_vecs, vocab

def load_transformer(language):

    trans_char_embeddings = pd.read_csv("transformer_embeddings/trans_char_embeddings-"+language+"_3000000_100ep_2.txt", sep=',', quotechar='"', index_col="CHAR")

    if language == "ja":
        for index, row in trans_char_embeddings.iterrows():
            if regex.search(r'\p{IsHan}|\p{IsKatakana}', index):
                trans_char_embeddings.drop(index, inplace=True)

    char_vector = trans_char_embeddings.to_numpy()
    char_vecs = np.array([x for x in char_vector])
    vocab = trans_char_embeddings.index
    return trans_char_embeddings, char_vecs, vocab




def load_color(language):
    if language == "en":
        language = "English"
    elif language == "es":
        language = "Spanish"
    elif language == "nl":
        language = "Dutch"
    elif language == "ko":
        language = "Korean"
    elif language == "ja":
        language = "Japanese"
    else:
        sys.exit("UNKNOWN LANGUAGE: ", language)

    synesthesia_data = pd.read_csv('color_embeddings/syn_data_Apaper.csv')
    language_data = synesthesia_data.loc[synesthesia_data['Language'] == language]
    return language_data, None, None
