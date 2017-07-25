import random
import sys

import numpy as np
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.models import Sequential
from keras.optimizers import RMSprop

from utils.dataset_utils import read_songs

sequence_length = 60


def train(data_paths):
    # sess = tf.Session()
    # K.set_session(sess)

    text = '\n'.join([read_songs(path) for path in data_paths]).replace('\n\n', '\n').lower()

    chars = sorted(list(set(text)))
    print('total chars:', len(chars))

    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    sentences = []
    next_chars = []

    for i in range(0, len(text) - sequence_length):
        sentences.append(text[i: i + sequence_length])
        next_chars.append(text[i + sequence_length])
    print('nb sequences:', len(sentences))

    print('Data model...')
    X = np.zeros((len(sentences), sequence_length, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    print('Build model...')
    model = Sequential()
    model.add(LSTM(512, input_shape=(sequence_length, len(chars)), return_sequences=True))
    model.add(LSTM(512, input_shape=(sequence_length, len(chars))))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    def sample(preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    # train the model, output generated text after each iteration
    for iteration in range(1, 60):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        model.fit(X, y,
                  batch_size=128,
                  epochs=1)

        start_index = random.randint(0, len(text) - sequence_length - 1)

        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print()
            print('----- diversity:', diversity)

            generated = ''
            sentence = text[start_index: start_index + sequence_length]
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)

            for i in range(400):
                x = np.zeros((1, sequence_length, len(chars)))
                for t, char in enumerate(sentence):
                    x[0, t, char_indices[char]] = 1.

                preds = model.predict(x, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()

train(['../data/lyrics/dragonforce', '../data/lyrics/stratovarius'])