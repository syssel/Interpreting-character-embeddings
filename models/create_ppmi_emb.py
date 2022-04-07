import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from collections import defaultdict

def get_ngrams(word, index, n):
    word = word[:]
    word[index] = "_"

    word=list("#"*(n-1))+word+list((n-1)*"#")
    index+=n-1
    return ("-".join(word[index-n+i:index+i]) for i in range(1,n+1))


def create_ppmi_matrix(data, n, n_words):
    context_counts = defaultdict(lambda:defaultdict(lambda:0))

    seen = set()
    vocab = set()
    print("Collecting and counting n-grams")
    with open(data) as f:
        for line in tqdm(f, total=n_words):

            if not line: continue

            word = line.strip().split(" ")

            if tuple(word) in seen: continue
            seen.add(tuple(word))

            for index in range(len(word)):
                target = word[index]
                vocab.add(target)

                for context in get_ngrams(word, index, n):
                    context_counts[context][target]+=1
    del seen
    print("Creating count matrix")

    vocab = list(vocab)
    contexts = list(context_counts.keys())
    count_matrix = np.zeros((len(vocab), len(contexts)))

    for context, vocab_items in tqdm(context_counts.items(),total=len(contexts)):
        for item, count in vocab_items.items():
            i = vocab.index(item)
            j = contexts.index(context)
            count_matrix[i][j] = count
    
    print("Shape:", count_matrix.shape)

    total = np.sum(count_matrix)
    col_total = np.sum(count_matrix, axis=0)
    row_total = np.sum(count_matrix, axis=1)

    expected = np.outer(row_total, col_total) / total

    ppmi_matrix = count_matrix/expected

    with np.errstate(divide='ignore'):
        ppmi_matrix = np.log(ppmi_matrix)
    
    ppmi_matrix[np.isinf(ppmi_matrix)] = 0.0
    ppmi_matrix[ppmi_matrix < 0] = 0.0
    
    return pd.DataFrame(ppmi_matrix, columns=contexts, index=vocab)


N = 3
max_chars = 3000000

SOURCE_DIR = "../preprocessed_data/"
OUT_DIR = "../ppmi_embeddings/"
    
for entry in os.scandir(SOURCE_DIR):
    if entry.name.startswith("train_") and entry.name.endswith(".txt"):
        try:
            lang = Path(entry.name).stem[entry.name.index("_")+1:]


            # Create file to train PPMI
            print("[{}]\tPreparing data to train PPMI embeddings".format(lang))
            ppmi_file = entry.path+".ppmi"
            
            w = open(ppmi_file, "w")
            with open(entry.path) as f:
                n_words = 0
                n_chars = 0
                for i, line in enumerate(f):
                    line=line.lower()
                    n_chars+=len(line)
                    words = [" ".join(list(word)) for word in line.strip().split()]
                    n_words+=len(words)
                    if i == 0:
                        print("Sample output:")
                        print("\n".join(words))
                    w.write("\n".join(words)+"\n")
                    if n_chars >= max_chars: break

                print()
                print("Listed '{}' words in total".format(n_words))
                print("And '{}' characters".format(n_chars))                
            print()
            print("[{}]\tTraining PPMI embeddings".format(lang))
            
            model_name = OUT_DIR+lang
            df = create_ppmi_matrix(ppmi_file, N, n_words)
            df.to_csv(model_name+".txt", sep=",", index=True, index_label="CHAR", header=True)

            print("[{}]\tOut: {}".format(lang, model_name+".txt"))
            print()
            w.close()

        except KeyboardInterrupt:
            print("Aborting...")
            os.remove(ppmi_file)

