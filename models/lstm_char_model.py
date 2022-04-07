from numpy.random import seed
seed(123)
from tensorflow import random
random.set_seed(123)

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from keras import backend as K
import numpy as np
import random
import io
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, History
from keras.models import load_model
import math

#LSTM character model
# Original source: https://keras.io/examples/generative/lstm_character_level_text_generation/#build-the-model-a-single-lstm-layer

def perplexity(y_true, y_pred):
    perplexity = 2 ** K.sparse_categorical_crossentropy(y_true, y_pred)
    return perplexity

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


language = "es"
max_chars = 3000000
result_file = open("lstm_models/result_log.txt", "a")
infile = "./preprocessed_data/train_"+language+".txt"
#infile = "./preprocessed_data/train_ja-kanji2hira.txt"
print(infile)

with io.open(infile, encoding="utf-8") as f:
    text = f.read().lower()
text = text.replace("\n", " ")  # remove newlines chars for nicer display
text = text[:max_chars]
print("Total chars:", len(text))

chars = sorted(list(set(text)))
print("Unique chars:", len(chars))
print(chars)
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

x = np.zeros((len(sentences), maxlen), dtype=np.int32)
y = np.zeros((len(sentences)), dtype=np.int32)
for i, sentence in enumerate(sentences):
    x[i] = [char_indices[char] for char in sentence]
    y[i] = char_indices[next_chars[i]]

x = tf.keras.utils.to_categorical(x, len(chars))


# model parameters
embedding_dim = 256 #256 for biLSTM
learning_rate = 0.01
epochs = 100
batch_size = 128
bidirectional = True
no_layers = 2

inputs = layers.Input(shape=(maxlen,len(chars)))

if bidirectional:
    lstm = layers.Bidirectional(layers.LSTM(int(embedding_dim/2)))(inputs)
    if no_layers == 2:
        lstm = layers.Bidirectional(layers.LSTM(int(embedding_dim/2), return_sequences=True))(inputs)
        lstm = layers.Bidirectional(layers.LSTM(int(embedding_dim/2)))(lstm)
else:
    lstm = layers.LSTM(embedding_dim)(inputs)
    if no_layers == 2:
        lstm = layers.LSTM(embedding_dim, return_sequences=True)(inputs)
        lstm = layers.LSTM(embedding_dim)(lstm)
outputs = layers.Dense(len(chars), activation="softmax")(lstm)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()#from_logits=True) # from_logits=True

model = keras.Model(inputs=inputs, outputs=outputs)
print(model.summary())
model.compile(loss=loss_fn, optimizer=optimizer, metrics=["accuracy", perplexity])


# define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
model_checkpoint = ModelCheckpoint('lstm_models/best_model-'+language+"_"+str(max_chars)+"_"+str(epochs)+"ep_"+str(no_layers)+'lay.h5', monitor='val_loss', verbose=1, save_best_only=True)
history = History()

x = model.fit(x, y, validation_split=0.1, batch_size=batch_size, epochs=epochs, callbacks=[early_stopping, model_checkpoint, history])

# log results in file
print(language, max_chars, epochs, batch_size, embedding_dim, learning_rate, min(history.history['loss']), min(history.history['val_loss']), min(history.history['perplexity']), min(history.history['val_perplexity']), bidirectional, no_layers, file=result_file)

# load best model
saved_model = load_model('lstm_models/best_model-'+language+"_"+str(max_chars)+"_"+str(epochs)+"ep_"+str(no_layers)+'lay.h5', custom_objects={"perplexity": perplexity})

# sample from model to generate text
print()
print("Generating text after traning:")

start_index = random.randint(0, len(text) - maxlen - 1)


for diversity in [0.5, 1.0]:
    print("...Diversity:", diversity)

    generated = ""
    sentence = text[start_index : start_index + maxlen]
    print('...Generating with seed: "' + sentence + '"')

    for i in range(400):
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1.0
        preds = saved_model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]
        sentence = sentence[1:] + next_char
        generated += next_char

    print("...Generated: ", generated)
    print()


# save LSTM embeddings from best model
hidden_lstm_weights = saved_model.layers[-1].get_weights()[0]
hidden_lstm_weights = np.transpose(hidden_lstm_weights)
print("HIDDEN-TO_OUTPUT WEIGHTS SHAPE:", hidden_lstm_weights.shape)
if bidirectional is True:
    embedding_file = open("lstm_embeddings/bilstm_char_embeddings-"+language+"_"+str(max_chars)+"_"+str(epochs)+"ep_"+str(no_layers)+"lay.txt", "w")
else:
    embedding_file = open("lstm_embeddings/lstm_char_embeddings-"+language+"_"+str(max_chars)+"_"+str(epochs)+"ep_"+str(no_layers)+"lay.txt", "w")
print("CHAR," + ",".join(map(str, list(range(embedding_dim)))), file=embedding_file)
for idx, embedding in enumerate(hidden_lstm_weights):
    print('"' + indices_char[idx] + '",'+ ",".join(map(str, embedding)), file=embedding_file)
