import os
import random as rnd
import string

import numpy as np

from batchers.Batcher import Batcher
from utils.dataset_utils import read_songs

VOCABULARY = string.ascii_letters + ',.?!\'\n '
VOCABULARY_LOOKUP = dict((char, i) for i, char in enumerate(VOCABULARY))
VOCABULARY_SIZE = len(VOCABULARY)


class CharacterBatcher(Batcher):
    vocabulary_size = VOCABULARY_SIZE
    cursor = 0

    def __init__(self, data_path: str, random: bool = False, batch_size: int = 20, sequence_length: int = 8) -> None:
        self.random = random
        data_raw = read_songs(data_path).replace('\n\n', '\n')
        data_raw = filter(lambda char: char in VOCABULARY_LOOKUP, data_raw)
        data = list(map(lambda char: VOCABULARY_LOOKUP[char], data_raw))  # translate to indices

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
        return "".join(map(lambda a: VOCABULARY[a], c))

    def encode_text(self, s: str) -> list:
        return list(map(lambda c: VOCABULARY_LOOKUP[c], s))

    def get_vocabulary_size(self):
        return VOCABULARY_SIZE
