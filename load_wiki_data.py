import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import string
import re
import os
import regex
import jamotools
import pykakasi

kks = pykakasi.kakasi()

def download_data(language):
    """download dataset from TF Hub"""
    test_data, val_data, train_data = tfds.load(name="wiki40b/"+language, split=["test", "validation", "train"], data_dir="./tensorflow_datasets/")
    print("DOWNLOAD COMPLETED.")

def preprocess_data(language, set):
    """load binary files with wikipedia data and preprocess for character model training
    preprocessing steps: remove informational tags, filter special characters and normalize space characters
    save a plain text"""

    directory = "./tensorflow_datasets/wiki40b/" + language + "/1.3.0/"

    text_data = open("./preprocessed_data/"+set+"_"+language+"-kanji2hira.txt", "w")
    count_chars = 0

    for filename in os.listdir(directory):
        if set in filename:
            print(filename)

            infile = open(directory+filename,"rb")
            train_data = infile.readlines()
            train_data = [l.decode("utf-8", errors="replace") for l in train_data]
            #print(len(train_data))
            infile.close()

            info_lines = ["_START_ARTICLE_", "_START_SECTION_", "_START_PARAGRAPH_"]

            text = [l for idx, l in enumerate(train_data) if train_data[idx-1].strip() in info_lines]

            # for Dutch and English
            if language == "nl" or language == "en":
                special_chars = re.compile("[^A-Za-z0-9\s\.,;:?!+\-%&#']")
            elif language == "es":
                special_chars = re.compile('[^A-Za-z0-9\s\.,;:?!+\-%&#áéíóúÁÉÍÓÚñÑ¿¡]')
            elif language == "ja" or language == "ko":
                special_chars = re.compile('[0-9\s\.,;:?!+\-%&#。、]')

            spaces = re.compile('\s')

            for l in text:
                l = l.replace("_NEWLINE_", "\n")
                if language == "ko":
                    for char in l:
                        if (not regex.search(r'\p{IsHangul}', char)) and (not special_chars.search(char)):
                            l = l.replace(char, "€")
                        #split Korean characters into Jamos
                        #else:
                            #split_char = jamotools.split_syllables(char)
                            #l = l.replace(char, split_char)
                elif language == "ja":
                    for char in l:
                        # for original submission - remove kanji
                        #if (not regex.search(r'\p{IsHiragana}|\p{IsKatakana}', char)) and (not special_chars.search(char)): #\p{IsHan}
                        #    l = l.replace(char, "€")

                        # new: replace Kanjis with hiragana transcription to have a full syllabic alphabet
                        if (not regex.search(r'\p{IsHiragana}|\p{IsKatakana}', char)) and (not special_chars.search(char)): #\p{IsHan}

                            try:
                                converted = kks.convert(char)
                                hira_chars = converted[0]['hira']
                                l = l.replace(char, hira_chars)
                            except IndexError:
                                print(char)
                                continue
                elif language == "es":
                    l = l.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u").replace("Á", "A").replace("É", "E").replace("Í", "I").replace("Ó", "O").replace("Ú", "U")
                    l = re.sub(special_chars, "€", l)
                else:
                    l = re.sub(special_chars, "€", l)
                l = re.sub(spaces, " ", l)

                # replace multiple € (unkown chars) with only one
                unkown_chars = re.compile('€+')
                l = re.sub(unkown_chars, "€", l)

                count_chars += len(l)

                # set cut off at 10 mio. characters for training data and 9 mio. for test/val data
                if set == "train":
                    if count_chars <= 100000000:
                        print(l, file = text_data)
                    else:
                        break
                elif set == "test" or set == "validation":
                    if count_chars <= 9000000:
                        print(l, file = text_data)
                    else:
                        break
            #print(count_chars)

    print(count_chars, "for lang:", language, set)


# ngram model: https://github.com/connormayer/distributional_learning

# RNN LM: https://github.com/syssel/examples/tree/master/word_language_model

def main():
    lang_code = "ja" #ko, nl, es, ja, en
    #download_data(lang_code)
    for set in [ "test", "validation"]: #"train"],
        preprocess_data(lang_code, set)


if __name__ == "__main__":
    # execute only if run as a script
    main()
