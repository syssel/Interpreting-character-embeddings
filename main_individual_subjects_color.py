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

color_type = "individual_subjs" # or average
subjects = ['JP001','JP002','JP004','JP005','JP006','JP007','JP008','JP011','JP012','JP013','JP014','JP016','JP017','JP018','JP019','JP021','JP024','JP025','JP034','JP037','JP063','JP065','JP066','JP068','JP069','JP082','JP087']
#subjects = ['1','2','3','4','5','6','7','8','9','10','11','12','13'] #ko
#subjects = ['1583','2182','2757','2789','3600','3852','4378','4541','5084','5615','6886','7472','7531','7552','7700','7772','8551','8587','8902','9835','10504','12329','13389','14015','15596','16078','16483','16630','16740','18170','18671','19089','22307','25132','25766','25770','25785','26764','33961','33990','34578','34815','34823','34877','34885','37714','38314'] #en
#subjects = ['11419','11442','12682','14595','14599','20538','20793','21179','21695','21935','22049','22880','26723','28090','28095','41068','41490','41510','42426','42527','42885','43015','43701','44040','117930','120874','120875','122019','145250','154075','156641','157074'] #es
#subjects = ['13602','17790','37016','46830','46831','46855','46856','46857','46860','46863','46864','46876','46878','46884','46887','46916','46918','46929','46930','46934','46966','47036','47052','47095','47200','47242','47286','47297','47679','47781','48111','48583','48636','48749','48760','48763','48764','48767','48791','48814','48884','48885','48941','48955','49011','49012','49075','49103','49141','49180','49312','49574','49577','49579','49645','50003','50130','50268','50519','52097','58539','58723','64159','67751','69029','83856','84097','85842','89283','98674','100125','104175','104290','104521','117535','127866','127868','127872','127887','127906','127940','128008','128010','128013','128015','128021','128046','128082','128182','128254','128316','128492','128969','129292','129632','940','130258','130264','130276','130310','130312','130313','130344','130366','130396','130410','130454','130576','130681','131213'] #nl

for subj in subjects:
    #print(subj)
    for lang in args.languages:
        #print("LANGUAGE: ", lang)

        correlations = np.zeros(shape=(len(args.embedding_types),len(args.embedding_types)))
        pvalues = np.zeros(shape=(len(args.embedding_types),len(args.embedding_types)))
        correlations_random_baseline = np.zeros(shape=(len(args.embedding_types),len(args.embedding_types)))
        pvalues_random_baseline = np.zeros(shape=(len(args.embedding_types),len(args.embedding_types)))

        # load embeddings
        for embedding_combi in list(itertools.combinations(args.embedding_types, 2)):

            embeddings = {}
            #print("\nCOMBINATION: ", embedding_combi)

            for idx, embedding_type in enumerate(embedding_combi):
                if embedding_type == "shape":
                    distances, vocabulary = calculate_shape_distances(lang)
                    embeddings["embeddings{0}".format(idx)] = [distances, vocabulary]
                elif embedding_type == "ipa":
                    distances, vocabulary = calculate_ipa_distances(lang)
                    embeddings["embeddings{0}".format(idx)] = [distances, vocabulary]
                elif embedding_type == "color":
                    distances, vocabulary = calculate_color_distances(lang, type=color_type)
                    distances = distances.loc[distances['subject'] == subj]
                    distances = distances.iloc[:, :-1]
                    distances = distances.dropna(axis=1)
                    #print(distances)
                    embeddings["embeddings{0}".format(idx)] = [distances, vocabulary]
                elif embedding_type == "ppmi":
                    distances, vocabulary = calculate_ppmi_distances(lang)
                    embeddings["embeddings{0}".format(idx)] = [distances, vocabulary]
                elif embedding_type == "lstm":
                    distances, vocabulary = calculate_lstm_distances(lang)
                    embeddings["embeddings{0}".format(idx)] = [distances, vocabulary]
                elif embedding_type == "transformer":
                    distances, vocabulary = calculate_transformer_distances(lang)
                    embeddings["embeddings{0}".format(idx)] = [distances, vocabulary]
                else:
                    sys.exit("Embedding type unknown!")

            try:
                shared_vocabulary = np.intersect1d(embeddings["embeddings0"][1],embeddings["embeddings1"][1])
                distances0 = embeddings["embeddings0"][0]
                distances1 = embeddings["embeddings1"][0]
                shared_distances = distances0.columns.intersection(distances1.columns)
                print (embedding_combi[0], embedding_combi[1], subj, pearsonr(distances0[shared_distances].iloc[0],distances1[shared_distances].iloc[0]))
            except:
                print("except")
                continue
