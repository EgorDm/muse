import math, time
import numpy as np
import tensorflow as tf
from utils.utils import mkdir_or_not


class BaseTrainer:
    def __init__(self, batcher, validation_batcher, model, settings):

        self.batcher = batcher
        self.validation_batcher = validation_batcher
        self.model = model

        timestamp = str(math.trunc(time.time()))
        self.model_name = '{}_{}'.format(settings.name, timestamp)

        mkdir_or_not(settings.log_dir)
        mkdir_or_not(settings.save_dir)

        self.save_path = settings.save_dir

        self.summary_writer = tf.summary.FileWriter('{}/{}-training'.format(settings.log_dir, self.model_name))
        self.validation_writer = tf.summary.FileWriter('{}/{}-validation'.format(settings.log_dir, self.model_name))

        self.saver = tf.train.Saver(max_to_keep=1000)

        self.lr = settings.lr
        self.kprob = settings.kprob
        self.prime = settings.prime
        self.display_freq = settings.display_freq

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
