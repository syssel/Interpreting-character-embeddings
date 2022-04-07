from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
from sklearn import manifold
from sklearn.manifold import TSNE
import pandas as pd



def hierarchical_cluster(X, labelList, language, type, subject=None):

    plt.figure()

    method = 'single'
    linked = linkage(X, method, metric='euclidean', optimal_ordering=True)

    dendrogram(linked,
                orientation='top',
                labels=labelList,
                distance_sort='descending')
    plt.tick_params(left = False,labelleft = False)

    if language == "Korean" or language == "Japanese" or language == "ko" or language == "ja":
        fprop = fm.FontProperties(fname='fonts/NotoSansKR-Light.otf')
        plt.xticks(fontproperties=fprop)
    if subject:
        plt.title("Subject:" + str(subject))
        plt.savefig('clustering/hierarchical-clusters-'+type+"-"+language+subject+'.pdf')
        print("Cluster plot saved to ", 'clustering/hierarchical-clusters-'+type+"-"+language+subject+'.pdf')
    else:
        plt.savefig('clustering/hierarchical-clusters-'+type+"-"+language+'.pdf')
        print("Cluster plot saved to ", 'clustering/hierarchical-clusters-'+type+"-"+language+'.pdf')
    plt.close()

def tsne_clustering(X, labelList, language, type, subject=None):

    """Use tSNE to reduce the dimensions of the embeddings into two-dimensional space, then plot the resulting clusters."""

    tsne = manifold.TSNE(n_components=2, init='random',
                         random_state=0)

    Y = tsne.fit_transform(X)
    #print(Y)

    tsne_df_scale = pd.DataFrame(Y, columns=['tsne1', 'tsne2'])

    fig, ax = plt.subplots()
    #ax.scatter(tsne_df_scale.iloc[:,0],tsne_df_scale.iloc[:,1], alpha=0.25, facecolor='blue')

    alphabet_simple = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    korean_simple = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ', 'ㅏ', 'ㅐ', 'ㅑ', 'ㅓ','ㅔ' ,'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
    japanese_simple = ['あ', 'い', 'う', 'え', 'お', 'を', 'か', 'き', 'く', 'け', 'こ', 'さ', 'し', 'す', 'せ', 'そ', 'た', 'ち', 'つ',
     'て', 'と', 'な', 'に', 'ぬ', 'ね', 'の', 'は', 'ひ', 'ふ', 'へ', 'ほ', 'ま', 'み', 'む', 'め', 'も', 'や',
     'ゆ', 'よ', 'ら', 'り', 'る', 'れ', 'ろ', 'わ', 'ん']

    if language == "Korean" or language == "ko":
        for i, (x, y) in enumerate(zip(tsne_df_scale.iloc[:,0],tsne_df_scale.iloc[:,1])):
            if labelList[i] in korean_simple:
                if labelList[i] in korean_simple[:14]:
                    ax.scatter(x,y, facecolor='#44A2C4', label='consonant', s=3)
                    fprop = fm.FontProperties(fname='fonts/NotoSansKR-Light.otf')
                    ax.annotate(labelList[i], (tsne_df_scale.iloc[i,0], tsne_df_scale.iloc[i,1]), fontproperties=fprop, fontsize=22, weight='bold', color='#44A2C4')
                elif labelList[i] in korean_simple[14:]:
                    ax.scatter(x,y, facecolor='#A31D20', label='vowel', s=3)
                    fprop = fm.FontProperties(fname='fonts/NotoSansKR-Light.otf')
                    ax.annotate(labelList[i], (tsne_df_scale.iloc[i,0], tsne_df_scale.iloc[i,1]), fontproperties=fprop, fontsize=22, weight='bold', color='#A31D20')
    elif language == "Japanese" or language == "ja":
        # todo: add pronunciation to labels
        for i, (x, y) in enumerate(zip(tsne_df_scale.iloc[:,0],tsne_df_scale.iloc[:,1])):
            if labelList[i] in japanese_simple:
                if labelList[i] in japanese_simple[:6]:
                    ax.scatter(x,y, facecolor='#A31D20', label='singular vowels', s=3)
                    fprop = fm.FontProperties(fname='fonts/NotoSansKR-Light.otf')
                    ax.annotate(labelList[i], (tsne_df_scale.iloc[i,0], tsne_df_scale.iloc[i,1]), fontproperties=fprop, fontsize=22, weight='bold', color='#A31D20')
                elif labelList[i] in japanese_simple[6:-1]:
                    ax.scatter(x,y, facecolor='#92D050', label='consonant-vowel unions', s=3)
                    fprop = fm.FontProperties(fname='fonts/NotoSansKR-Light.otf')
                    ax.annotate(labelList[i], (tsne_df_scale.iloc[i,0], tsne_df_scale.iloc[i,1]), fontproperties=fprop, fontsize=22, weight='bold', color='#92D050')
                elif labelList[i] == japanese_simple[-1]:
                    ax.scatter(x,y, facecolor='#44A2C4', label='singular consonant', s=3)
                    fprop = fm.FontProperties(fname='fonts/NotoSansKR-Light.otf')
                    ax.annotate(labelList[i], (tsne_df_scale.iloc[i,0], tsne_df_scale.iloc[i,1]), fontproperties=fprop, fontsize=22, weight='bold', color='#44A2C4')
    else:
        for i, (x, y) in enumerate(zip(tsne_df_scale.iloc[:,0],tsne_df_scale.iloc[:,1])):
            if labelList[i] in alphabet_simple:
                if labelList[i] in ['a', 'e', 'i', 'o', 'u']:
                    ax.scatter(x,y, facecolor='#A31D20', label='vowel', s=3)
                    ax.annotate(labelList[i], (tsne_df_scale.iloc[i,0], tsne_df_scale.iloc[i,1]), fontsize=22, weight='bold', color='#A31D20')
                elif labelList[i] in ['.', ',', ';', ':', '!', '#', '+', '?', '%', '&']:
                    ax.scatter(x,y, facecolor='#92D050', label='punctuation', s=3)
                    ax.annotate(labelList[i], (tsne_df_scale.iloc[i,0], tsne_df_scale.iloc[i,1]), fontsize=22, weight='bold', color='#92D050')
                elif labelList[i] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    ax.scatter(x,y, facecolor='#FFB14C', label='digits', s=3)
                    ax.annotate(labelList[i], (tsne_df_scale.iloc[i,0], tsne_df_scale.iloc[i,1]), fontsize=22, weight='bold', color='#FFB14C')
                else:
                    ax.scatter(x,y, facecolor='#44A2C4', label='consonant', s=3)
                    ax.annotate(labelList[i], (tsne_df_scale.iloc[i,0], tsne_df_scale.iloc[i,1]), fontsize=22, weight='bold', color='#44A2C4')

    #plt.xlabel('tsne1')
    #plt.ylabel('tsne2')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.title(language + " " + type, fontsize=22)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    lgnd = plt.legend(by_label.values(), by_label.keys(), fontsize=12)
    lgnd.legendHandles[0]._sizes = [30]
    lgnd.legendHandles[1]._sizes = [30]
    if language == "ja":
        lgnd.legendHandles[2]._sizes = [30]
    plt.savefig('clustering/TSNEclusters-'+type+"-"+language+'-forSlides.png')
    #plt.show()
    plt.close()
