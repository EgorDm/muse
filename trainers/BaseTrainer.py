import math, time
import numpy as np
import tensorflow as tf
from utils.utils import mkdir_or_not


class BaseTrainer:
    def __init__(self, batcher, validation_batcher, model, model_name, log_path, save_path):
        self.batcher = batcher
        self.validation_batcher = validation_batcher
        self.model = model

        timestamp = str(math.trunc(time.time()))
        self.model_name = '{}_{}'.format(model_name, timestamp)

        mkdir_or_not(log_path)
        mkdir_or_not(save_path)

        self.save_path = save_path

        self.summary_writer = tf.summary.FileWriter('{}/{}-training'.format(log_path, self.model_name))
        self.validation_writer = tf.summary.FileWriter('{}/{}-validation'.format(log_path, self.model_name))

        self.saver = tf.train.Saver(max_to_keep=1000)

    def train(self, epochs: int):
        pass

    def _train_step(self, state: np.ndarray) -> np.ndarray:
        """
        :type state: np.ndarray
        :rtype np.ndarray
        :param state:
        :return:
        """
        pass

    def _train_log(self, inputs: np.ndarray, labels: np.ndarray, state: np.ndarray):
        pass

    def _validate(self):
        pass
