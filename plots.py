from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_correlation_heatmap(correlations, column_labels, lang, pvalues):

    lang_keys = {
        "en":"English",
        "ja":"Japanese",
        "es":"Spanish",
        "nl":"Dutch",
        "ko":"Korean",
        "ko-syll":"Korean (syllables)",
        "RANDOM": "RANDOM",
    }

    df = pd.DataFrame(correlations, columns = column_labels)

    mask = np.zeros_like(df)
    mask[np.tril_indices_from(mask)] = True
    cmap = sns.diverging_palette(250, 15, as_cmap=True)


    # mark significant correlations
    labels = np.zeros_like(correlations, dtype=object)
    for ir,row in enumerate(correlations):
        for iv, value in enumerate(row):
            #significante threshold after Bonferroni correction (threshold/5) because 5 languages
            # 3 hypotheses for each type of model (ppmi, lstm, transformer)?
            if pvalues[ir, iv] < 0.01/5 and pvalues[ir, iv]!=1:
                labels[ir, iv] = "{:.2f}".format(value)+"**"
            # significance threshold normal
            elif pvalues[ir, iv] < 0.01 and pvalues[ir, iv]!=1:
                labels[ir, iv] = "{:.2f}".format(value)+"*"
            else:
                labels[ir, iv] = "{:.2f}".format(value)

    # remove redundant first column and last row for nicer plots
    df = df.iloc[:-1 , 1:]
    mask = mask[:-1 , 1:]
    labels = labels[:-1 , 1:]

    fig, ax = plt.subplots(figsize=(11, 9))
    ax = sns.heatmap(df, annot=labels, yticklabels=column_labels[:-1], mask=mask, cmap=cmap, vmin=-1, vmax=1, fmt="", annot_kws={"fontsize":20})
    ax.xaxis.tick_top() # x axis on top
    ax.xaxis.set_label_position('top')
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 20)
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 20)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    plt.title(lang_keys[lang], fontsize=24)
    plt.savefig("plots/"+lang+"-correlations-bonferroni-forSlides.pdf")
