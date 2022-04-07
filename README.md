# Character embedding analysis

This is the code used for the following paper:  
Sidsel Bolsen, Manex Agirrezabal & Nora Hollenstein. "Interpreting Character Embeddings With Perceptual Representations: The Case of Shape, Sound, and Color". To appear in _ACL 2022_.

## Main script
Example usage:
`python main.py --languages en es nl --embedding_types lstm color`  
This calculates the correlations between LSTM character embeddings and synesthesia character representations for English, Spanish and Dutch. 

There is a folder for each type of embedding (PPMI, LSTM, biLSTM, transformer, sound, color, shape).  
The languages includes are English (en), Spanish (es), Dutch (nl), Korean (ko) and Japanese (ja).

## Helper scripts
- `load_wiki_data.py` Preprocessing of wikipedia dataset for all 5 languages
- `distances.py` Loads embeddings and calculates distances between all combinations of character embeddings
- `hierarchical_clustering.py` Plots clusters of character embeddings
