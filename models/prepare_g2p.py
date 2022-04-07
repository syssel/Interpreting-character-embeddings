import sys
import os
import jamotools

indir = "g2p/wikipron/"
outdir = "g2p/data/"
directory = os.fsencode(indir)
    
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if not filename.endswith(".tsv"): continue

    print("Reading {}...".format(indir+filename))
    lang_prefix = filename[:filename.index("_")]

    outfile = open(outdir+lang_prefix+".tsv", "a", encoding="utf-8")
    if lang_prefix == "kor": 
        outfile_nojamos = open(outdir+lang_prefix+"_noJamos"+".tsv", "a", encoding="utf-8")

    print("\t Writing to {}".format(outdir+lang_prefix+".tsv"))
    with open(indir+filename, "r", encoding="utf-8") as f:
        for line in f:
            word, phones = line.strip().split("\t")

            if lang_prefix == "kor":
                outfile_nojamos.write(" ".join(list(word))+"\t"+phones+"\n")
                word = jamotools.split_syllables(word)
            
            outfile.write(" ".join(list(word))+"\t"+phones+"\n")
    print("\t Done")
    outfile.close()