

import warnings
import ipapy
from ipapy.ipachar import *

import numpy as np

import argparse
import logging
import warnings
from collections import defaultdict

import pandas as pd


class IPAEmbedding(object):

    def __init__(self):

        descriptors = {
            "global" : { # TODO: Derived global features could be interesting,
                        # i.e. a label <nasal> for both nasal consonants and nazalised vowels,
                        # or <front> for both front vowels and consonants
                "type": [d.canonical_label for d in DG_TYPES.descriptors]
            },

            "consonant" : {
                "voicing": [d.canonical_label for d in DG_C_VOICING.descriptors],
                "place"  : [d.canonical_label for d in DG_C_PLACE.descriptors],
                "manner" : [d.canonical_label for d in DG_C_MANNER.descriptors],
            },

            "vowel" : {
                "height"   : [d.canonical_label for d in DG_V_HEIGHT.descriptors],
                "backness" : [d.canonical_label for d in DG_V_BACKNESS.descriptors],
                "roundness": [d.canonical_label for d in DG_V_ROUNDNESS.descriptors],
            },

            "diacritic" : {
                "feature": [d.canonical_label for d in DG_DIACRITICS.descriptors],
            },

            "suprasegmental" : {
                "stress": [d.canonical_label for d in DG_S_STRESS.descriptors],
                "length": [d.canonical_label for d in DG_S_LENGTH.descriptors],
                "break" : [d.canonical_label for d in DG_S_BREAK.descriptors],
            },

            "tone" : {
                "level"  : [d.canonical_label for d in DG_T_LEVEL.descriptors],
                "contour": [d.canonical_label for d in DG_T_CONTOUR.descriptors],
                "global" : [d.canonical_label for d in DG_T_GLOBAL.descriptors],
            },

        }

        features = {}

        for d1, values in descriptors.items():
            for d2, labels in values.items():
                for label in labels:
                    if label in features:
                        warnings.warn("Label ({}:{}) already in features as <{}>".format(label, (d1, d2), features[label]))

                    features[label] = {
                        "descriptor_major": d1,
                        "descriptor_minor": d2,
                        "id": "_".join([d1, d2, label])
                    }
                    
        self.features = features
        self.feature_names = [features[f]["id"] for f in self.features]
    
    def __getitem__(self, s):
        description = self.get_description(s)

        if not description and len(s)>1:     # TODO: This is a naive way of combining IPA symbols.
                                             # e.g., we should only allow one global type (i.e. <consonant> should overrule <diacritic>)
            description = []
            for symbol in list(s):
                description += self.get_description(symbol)
            
            logging.debug("The feature set of '{}' is a combination of features from the symbols {}".format(s, list(s)))
        
        description = set(description)

        return [1 if feature in description else 0 for feature in self.features]
    
    def get_description(self, s):
        try:
            return ipapy.UNICODE_TO_IPA[s].canonical_representation.split()
        except KeyError:
            logging.debug("Could not retrieve an IPA reprentation of the segment '{}'".format(s))
            return []




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract phonological representation of characters using g2p alignments.')
    parser.add_argument('inputfile', type=str,
                        help="""Input file containing g2p alignments.
                                The input is expected to be formatted as outputted from the aligner m2m (https://github.com/letter-to-phoneme/m2m-aligner),
                                i.e. g|r|a|p:h|e|m|e\tɡ|ɹ|æ|f|iː|m|_""")
    parser.add_argument('outdest', type=str,
                        help="Where to save the embedding table.")
    parser.add_argument('--mode', type=str, choices=("most_frequent", "most_frequent_word_initial", "average", "average_word_initial"),
                        help='Manner of which the embeddings are derived.', default="most_frequent")
    parser.add_argument('--simplex', action='store_true',
                        help='If only simple graphs should be included, i.e. only consisting of one character.',)
    parser.add_argument('--debug', action='store_true',
                        help='Whether warnings should be included in the log.',)
                                                
    args = parser.parse_args()
    open(args.outdest+'.log', "w")
    logging.basicConfig(filename=args.outdest+'.log',level=logging.DEBUG if args.debug else logging.INFO)

    f = open(args.inputfile)
    g2p = g2p = defaultdict(lambda:defaultdict(lambda:0))

    for line in f:
        graphs, phones = line.strip().split("\t")
        graphs, phones = graphs.split("|"), phones.split("|")

        if args.mode.endswith("_word_initial"): 
            graphs, phones = graphs[:1], phones[:1]
        
        for g, p in zip(graphs, phones):
            if args.simplex and ":" in g: # Do not add if the graph is complex and we only want simplex
                continue

            if p == "_": # Do not include zero mappings (g->NULL)
                continue 

            g2p[g][p] += 1
    f.close()
    
    E = IPAEmbedding()


    embtable = []
    for graph, phones in g2p.items():

        sorted_by_freq = sorted(phones.items(), key=lambda p: p[1], reverse=True)

        if args.mode.startswith("most_frequent"):
            repr = E[sorted_by_freq[0][0]]
            logging.info('Writing the most frequent phone mapping for the graph "{}": "{}" counting {} occurrences'.format(graph, sorted_by_freq[0][0], sorted_by_freq[0][1]))
        else:
            logging.info('Writing the phone mapping for the graph "{}" as the average of "{}"'.format(graph, sorted_by_freq))
            repr = np.array(E[""])
            occurrences = 0
            for phone, count in sorted_by_freq:
                occurrences +=count
                repr += np.array(E[phone]) * count
            repr = repr/occurrences

        embtable.append(repr)
    
    df = pd.DataFrame(embtable, columns=E.feature_names, index=g2p.keys())
    df.to_csv(args.outdest+".txt", sep=",", header=True, index=True, index_label="CHAR")
