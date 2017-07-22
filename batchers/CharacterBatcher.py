import os
import random as rnd
import string

import numpy as np

from batchers.Batcher import Batcher
from utils.dataset_utils import read_songs

SPEC_CHARS = ',.?!\'\n '


class CharacterBatcher(Batcher):
    cursor = 0

    def __init__(self, data_paths: list, lowercase: bool = True, random: bool = False, batch_size: int = 20,
                 sequence_length: int = 8) -> None:
        self.lowercase = lowercase
        self.random = random
        self._vocabulary, self._vocabulary_lookup, self._vocabulary_size = self._gen_vocabulary()

        data_raw = '\n'.join([read_songs(path) for path in data_paths]).replace('\n\n', '\n')
        if lowercase: data_raw = data_raw.lower()
        data_raw = filter(lambda char: char in self.get_vocabulary(), data_raw)
        data = list(map(lambda char: self.get_vocabulary_lookup()[char], data_raw))  # translate to indices

        self.data_length = len(data)
        self.epoch_size = self.data_length // (batch_size * sequence_length)

        super().__init__(data, batch_size, sequence_length)

    def get_sequence(self) -> (np.ndarray, np.ndarray):
        if self.random:
            return self._get_random_sequence()
        else:
            return self._get_next_sequence()

    def _get_random_sequence(self) -> (np.ndarray, np.ndarray):
        """
        :rtype: (np.ndarray, np.ndarray)
        :return:
        """
        while True:
            start = rnd.randint(0, len(self.data))
            end = start + self.sequence_length + 1
            if end >= len(self.data) or self.data[end - 1] == '\n': continue
            break

        inputs = np.array(self.data[start:end - 1])
        label = np.array(self.data[start + 1:end])
        return inputs, label

    def _get_next_sequence(self) -> (np.ndarray, np.ndarray):
        """
        :rtype: (np.ndarray, np.ndarray)
        :return:
        """
        if self.cursor + self.sequence_length + 1 >= self.data_length:
            self.cursor = 0
            self.epoch += 1
        start = self.cursor
        end = start + self.sequence_length + 1

        inputs = np.array(self.data[start:end - 1])
        label = np.array(self.data[start + 1:end])

        self.cursor += self.sequence_length + 1
        return inputs, label

    def decode_text(self, c: list) -> str:
        return "".join(map(lambda a: self.get_vocabulary()[a], c))

    def encode_text(self, s: str) -> list:
        return list(map(lambda c: self.get_vocabulary_lookup()[c], s))

    def _gen_vocabulary(self) -> (list, dict, int):
        if self.lowercase:
            vocab = string.ascii_lowercase + SPEC_CHARS
        else:
            vocab = string.ascii_letters + SPEC_CHARS
        lookup = dict((char, i) for i, char in enumerate(vocab))
        return vocab, lookup, len(vocab)
