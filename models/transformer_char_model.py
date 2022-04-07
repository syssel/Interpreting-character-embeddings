import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K
import numpy as np
import io
import random
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, History
from keras.models import load_model
import math


def perplexity(y_true, y_pred):
    perplexity = 2 ** K.sparse_categorical_crossentropy(y_true, y_pred)
    return perplexity

def sample(preds):
    """convert predictions to probabilities"""
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype("float64")
    #preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

class TransformerBlock(layers.Layer):
    def __init__(self, embedding_dim, num_heads, ff_dim, name=None, **kwargs):
        super(TransformerBlock, self).__init__(name="TransformerBlock")
        #self.k = k
        super(TransformerBlock, self).__init__(**kwargs)

        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embedding_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(0.1)
        self.dropout2 = layers.Dropout(0.1)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = {'embedding_dim': embedding_dim,
                'num_heads': num_heads,
                'ff_dim': ff_dim}
        base_config = super(TransformerBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TokenAndPositionEmbedding(layers.Layer):
    """Implement embedding layer
    Two seperate embedding layers, one for tokens, one for token index (positions)."""
    def __init__(self, maxlen, vocab_size, embedding_dim, name=None, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__(name="TokenAndPositionEmbedding")

        #self.k = k
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)

        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embedding_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = {#'token_emb': self.token_emb,
                  #'pos_emb': self.pos_emb,
                  'maxlen': maxlen,
                  'vocab_size': vocab_size,
                  'embedding_dim': embedding_dim}
        base_config = super(TokenAndPositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# Prepare dataset
result_file = open("transformer_models/result_log.txt", "a")

language = "ko"
max_chars = 3000000
##infile = "./preprocessed_data/train_"+language+".txt"
#infile = "./preprocessed_data/train_"+language+"-kanji2hira.txt"
infile = "./preprocessed_data/train_"+language+"-noJamos.txt"

with io.open(infile, encoding="utf-8") as f:
    text = f.read().lower()
text = text.replace("\n", " ")
text = text[:max_chars]
print("Total chars:", len(text))

chars = sorted(list(set(text)))
print("Unique chars:", len(chars))
print(chars)
vocab_size = len(chars)
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text into semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i : i + maxlen])
    next_chars.append(text[i + maxlen])
print("Number of sequences:", len(sentences))



x_train = np.zeros((len(sentences), maxlen), dtype=np.int32)
y_train = np.zeros((len(sentences)), dtype=np.int32)
for i, sentence in enumerate(sentences):
    x_train[i] = [char_indices[char] for char in sentence]
    y_train[i] = char_indices[next_chars[i]]

print(x_train[0])
print(y_train[0])

embedding_dim = 128  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 128  # Hidden layer size in feed forward network inside transformer
epochs=100
learning_rate=0.01
batch_size = 128

inputs = layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embedding_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embedding_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(vocab_size, activation="softmax")(x) # todo: add softmax activation?

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()


model = keras.Model(inputs=inputs, outputs=outputs)

# define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
model_checkpoint = ModelCheckpoint('transformer_models/best_model-'+language+"_"+str(max_chars)+"_"+str(epochs)+'ep.h5', monitor='val_loss', verbose=1, save_best_only=True)
history = History()

# train model
model.compile(optimizer, loss_fn, metrics=["accuracy", perplexity])
print(model.summary())
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[early_stopping, model_checkpoint])

# log results in file
print(language, max_chars, epochs, batch_size, embedding_dim, learning_rate, min(history.history['loss']), min(history.history['val_loss']), min(history.history['perplexity']), min(history.history['val_perplexity']), num_heads, file=result_file)

# load best model
#saved_model = load_model('transformer_models/best_model-'+language+"_"+str(max_chars)+"_"+str(epochs)+'ep.h5', custom_objects={"TokenAndPositionEmbedding": TokenAndPositionEmbedding})
#saved_model = load_model('transformer_models/best_model-'+language+"_"+str(max_chars)+"_"+str(epochs)+'ep.h5', custom_objects={"TokenAndPositionEmbedding": TokenAndPositionEmbedding, "TransformerBlock": TransformerBlock})

# sample from best model to generate text
print()
print("Generating text after traning:")
start_index = random.randint(0, len(text) - maxlen - 1)
generated = ""
sentence = text[start_index:start_index + maxlen]
sent_indices = [char_indices[char] for char in sentence]
print('...Generating with seed: "' + sentence + '"')

for i in range(200):
    x_pred = np.zeros((1, maxlen), dtype=np.int32)
    x_pred = [char_indices[char] for char in sentence]
    preds = model.predict(x_pred, verbose=0)[0]
    #ppl = perplexity(y_true, y_pred)
    next_index = sample(preds)
    next_char = indices_char[next_index]
    sentence = sentence[1:] + next_char
    generated += next_char

print("...Generated: ", generated)
print()

# save embeddings from model
hidden_trans_weights = model.layers[-1].get_weights()[0]
hidden_trans_weights = np.transpose(hidden_trans_weights)
print("HIDDEN-TO_OUTPUT WEIGHTS SHAPE:", hidden_trans_weights.shape)
embedding_file = open("transformer_embeddings/trans_char_embeddings-"+language+"_"+str(max_chars)+"_"+str(epochs)+"ep_"+str(num_heads)+".txt", "w")
print("CHAR," + ",".join(map(str, list(range(128)))), file=embedding_file)
for idx, embedding in enumerate(hidden_trans_weights):
    print('"' + indices_char[idx] + '",'+ ",".join(map(str, embedding)), file=embedding_file)

embedding_trans_weights = model.layers[1].get_weights()[0]
#embedding_trans_weights = np.transpose(embedding_trans_weights)
print("EMBEDING LAYER - TO_OUTPUT WEIGHTS SHAPE:", embedding_trans_weights.shape)
embedding_file = open("transformer_embeddings/trans_char_embeddings-embLayer-"+language+"_"+str(max_chars)+"_"+str(epochs)+"ep_"+str(num_heads)+".txt", "w")
print("CHAR," + ",".join(map(str, list(range(128)))), file=embedding_file)
for idx, embedding in enumerate(embedding_trans_weights):
    print('"' + indices_char[idx] + '",'+ ",".join(map(str, embedding)), file=embedding_file)
