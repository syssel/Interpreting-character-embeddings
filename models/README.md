# Character embedding models

## LSTM model

## Transformer model

## PPMI model

Creates PPMI matrices for all languages using the training files `../preprocessed_data/train_[language].txt`.

```bash
python create_ppmi_emb.py
```

## Color representations

## IPA representations


### Grapheme-to-phoneme alignment

#### Data
First, we get the pronunciation data from which we will retrieve IPA representations. We use the data from the SIGMORPHON 2020 Task 1: Multilingual Grapheme-to-Phoneme Conversion challenge:

```bash
# Create out folder
mkdir -p g2p/wikipron
mkdir g2p/data
mkdir -p g2p/m2m-output

# Download English (eng), Dutch (dut), Korean (kor), Spanish (spa), Japanese (jpn)
cd g2p/wikipron

## English
wget https://raw.githubusercontent.com/CUNY-CL/wikipron/master/data/scrape/tsv/eng_latn_uk_broad_filtered.tsv
wget https://raw.githubusercontent.com/CUNY-CL/wikipron/master/data/scrape/tsv/eng_latn_uk_narrow.tsv
wget https://raw.githubusercontent.com/CUNY-CL/wikipron/master/data/scrape/tsv/eng_latn_us_broad_filtered.tsv
wget https://raw.githubusercontent.com/CUNY-CL/wikipron/master/data/scrape/tsv/eng_latn_us_narrow.tsv

## Dutch
wget https://raw.githubusercontent.com/CUNY-CL/wikipron/master/data/scrape/tsv/dut_latn_broad_filtered.tsv
wget https://raw.githubusercontent.com/CUNY-CL/wikipron/master/data/scrape/tsv/dut_latn_narrow.tsv

## Korean
wget https://raw.githubusercontent.com/CUNY-CL/wikipron/master/data/scrape/tsv/kor_hang_narrow_filtered.tsv

## Spanish (Castilian)
wget https://raw.githubusercontent.com/CUNY-CL/wikipron/master/data/scrape/tsv/spa_latn_ca_broad_filtered.tsv
wget https://raw.githubusercontent.com/CUNY-CL/wikipron/master/data/scrape/tsv/spa_latn_ca_narrow.tsv

## Japanese (Hiragana)
wget https://raw.githubusercontent.com/CUNY-CL/wikipron/master/data/scrape/tsv/jpn_hira_narrow_filtered.tsv
```

#### Alignment
Second, we retrieve grapheme-to-phoneme alignments using the `m2m-aligner` library.


```bash
# Prepare data
python prepare_g2p.py
```

```bash
# Install m2m-aligner
mkdir lib
cd lib
git clone https://github.com/letter-to-phoneme/m2m-aligner.git
cd m2m-aligner
make
```

There are different settings in the m2m aligner that restricts the output, especially:
```
	--delX : allow deletion in the source side. (default:False)
	--delX : allow deletion in the target side. (default:False)
	--maxX <value> : the maximum size of sub-alignments in the source side. (default:2)
	--maxY <value> : the maximum size of sub-alignments in the target side. (default:2)
```

For English, we do not allow deletion from target to source side, i.e., no phoneme on the target side is realised as zero on the source side. This may be a simplifying assumption, but this avoids cases such as: `⟨s|l|_|i|m|e⟩` → `/s|l|a|ɪ|m|_/`, where the diphthong is incorrectly segmented.
On the other hand, we do allow deletions on the target side, as a grapheme may be realised as zero in the target, as the final -e in `⟨s|l|i|m|e⟩` → `/s|l|a;ɪ|m|_/`. This might, however, lead to false positives such as `⟨a|b|a|n|d|o|n⟩` → `/ə|b|æ|n|d|_|n̩/`.
Graphemes and phonemes consisting of more than two graphs/phones are rare. Thus we only allow for sub-alignments of size two.

For Dutch and Spanish, we apply the same parameters:

```
./m2m-aligner --delX --maxY 2 --maxX 2 --inputFile "g2p/data/dut.tsv"  --outputFile "g2p/m2m-output/dut_aligned.tsv" --alignerOut "g2p/m2m-output/dut.m"
./m2m-aligner --delX --maxY 2 --maxX 2 --inputFile "g2p/data/eng.tsv"  --outputFile "g2p/m2m-output/eng_aligned.tsv" --alignerOut "g2p/m2m-output/eng.m"
./m2m-aligner --delX --maxY 2 --maxX 2 --inputFile "g2p/data/spa.tsv"  --outputFile "m2m-output/spa_aligned.tsv" --alignerOut "m2m-output/spa.m"
```

As Japanese hiranga is a syllabic script, we should in principle not need sub-alignments on the target side but long vowels, for instance, are written as repeating symbols, e.g., `⟨あ;あ|い|う⟩` `/a̠ː|i|ɯ̟ᵝ/`. However, this is pretty rare, so we go on with not allowing sub-alignments on the target side. We will not allow deletions on the target side, either:

```
./m2m-aligner --maxY 2 --maxX 1 --inputFile "g2p/data/jpn.tsv"  --outputFile "g2p/m2m-output/jpn_aligned.tsv" --alignerOut "g2p/m2m-output/jpn.m"
```

Having split the Korean hangul letters into separate jamos, the correspondence between graphemes and phonemes should be close to one-to-one in most cases:

```
./m2m-aligner --maxY 1 --maxX 1 --inputFile "g2p/data/kor.tsv"  --outputFile "g2p/m2m-output/kor_aligned.tsv" --alignerOut "g2p/m2m-output/kor.m"
```

For the Korean setting without separating hangul letters into jamos, we chose the same setting as Japanese, but allow for syllables of three phonemes:

```
./m2m-aligner --maxY 3 --maxX 1 --inputFile "g2p/data/kor_noJamos.tsv"  --outputFile "g2p/m2m-output/kor_noJamos_aligned.tsv" --alignerOut "g2p/m2m-output/kor_noJamos.m"
```

##### Creating embeddings
Finally, we can create IPA embeddings using the script `create_phon_emb.py`:

```bash
python create_phon_emb.py -h
usage: create_phon_emb.py [-h] [--mode {most_frequent,most_frequent_word_initial,average,average_word_initial}] [--simplex] [--debug] inputfile outdest

Extract phonological representation of characters using g2p alignments.

positional arguments:
  inputfile             Input file containing g2p alignments. The input is expected to be formatted as  
                        outputted from the aligner m2m (https://github.com/letter-to-
                        phoneme/m2m-aligner), i.e. g|r|a|p:h|e|m|e ɡ|ɹ|æ|f|iː|m|_
  outdest               Where to save the embedding table.

optional arguments:
  -h, --help            show this help message and exit
  --mode {most_frequent,most_frequent_word_initial,average,average_word_initial}
                        Manner of which the embeddings are derived.
  --simplex             If only simple graphs should be included, i.e. only consisting of one character.
  --debug               Whether warnings should be included in the log.
```

In the paper we use the simplex flag (as we only treat characters and not graphemes) and the most_frequent strategy.

## Shape representations

