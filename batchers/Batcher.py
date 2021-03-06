from abc import abstractmethod, ABCMeta

import numpy as np


class Batcher(metaclass=ABCMeta):
    _vocabulary = []
    _vocabulary_lookup = {}
    _vocabulary_size = 0
    batch = 0
    epoch = 0

    def __init__(self, data: list, batch_size: int, sequence_length: int) -> None:
        self.data = data
        self.batch_size = batch_size
        self.sequence_length = sequence_length

    def get_batch(self) -> (np.ndarray, np.ndarray):
        """
        :rtype: (np.ndarray, np.ndarray)
        :return:
        """
        self.batch += 1
        inputs_vector = np.empty([self.batch_size, self.sequence_length], dtype=int)
        labels_vector = np.empty([self.batch_size, self.sequence_length], dtype=int)

        for i in range(self.batch_size):
            inputs, label = self.get_sequence()
            inputs_vector[i] = inputs
            labels_vector[i] = label

        return inputs_vector, labels_vector

    @abstractmethod
    def get_sequence(self) -> (np.ndarray, np.ndarray):
        """
        :rtype: (np.ndarray, np.ndarray)
        :return:
        """
        pass

    @abstractmethod
    def _gen_vocabulary(self) -> (list, dict, int):
        """
        :rtype: (list, dict, int)
        :return:
        """
        pass

    def get_vocabulary_size(self) -> int:
        return self._vocabulary_size

    def get_vocabulary(self) -> list:
        return self._vocabulary

    def get_vocabulary_lookup(self) -> dict:
        return self._vocabulary_lookup
