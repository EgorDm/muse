from trainers.BaseTrainer import BaseTrainer
from batchers.CharacterBatcher import VOCABULARY, VOCABULARY_LOOKUP
from utils.Logger import *
from utils.utils import *

import numpy as np
import tensorflow as tf


class MainTrainer(BaseTrainer):
    _sess = None

    def __init__(self, batcher, validation_batcher, model, settings):
        super().__init__(batcher, validation_batcher, model, settings)

    def train(self, epochs: int = 10):  # TODO continue where left off. Load state
        state = np.zeros([self.batcher.batch_size, self.model.nlayers * self.model.cell_size])
        init = tf.global_variables_initializer()
        self._sess = tf.Session()
        self._sess.run(init)

        progress = Progress(self.display_freq, size=111 + 2,
                            msg="Training on next " + str(self.display_freq) + " batches")

        while self.batcher.epoch <= epochs:
            state = self._train_step(state)

            if self.batcher.batch % self.display_freq == 0 and self.validation_batcher is not None:
                self._validate()

            if self.batcher.batch % (self.display_freq * 4) == 0:
                text = self.generate_text(self.prime)
                print_generated_text(text)

            if self.batcher.batch % (self.display_freq * 10) == 0:
                saved_file = self.saver.save(self._sess, '{}/{}'.format(self.save_path, self.model_name),
                                             global_step=self.batcher.batch)
                print("Saved file: " + saved_file)

            progress.step(reset=self.batcher.batch % self.display_freq == 0)

        print('Done with training')

    def _train_step(self, state: np.ndarray) -> np.ndarray:
        inputs, labels = self.batcher.get_batch()

        feed_dict = {self.model.X: inputs, self.model.Y_: labels, self.model.Hin: state, self.model.lr: self.lr,
                     self.model.pkeep: self.kprob, self.model.batch_size: self.batcher.batch_size}

        _, predictions, output_state = self._sess.run([self.model.optimizer, self.model.Y, self.model.H],
                                                      feed_dict=feed_dict)

        if self.batcher.batch % self.display_freq == 0:
            self._train_log(inputs, labels, state)

        return output_state

    def _train_log(self, inputs: np.ndarray, labels: np.ndarray, state: np.ndarray):
        feed_dict = {self.model.X: inputs, self.model.Y_: labels, self.model.Hin: state,
                     self.model.pkeep: 1.0, self.model.batch_size: self.batcher.batch_size}
        y, l, bl, acc, smm = self._sess.run(
            [self.model.Y, self.model.seqloss, self.model.batchloss, self.model.accuracy, self.model.summaries],
            feed_dict=feed_dict)
        print_learning_learned_comparison(self.batcher, inputs, y, l, bl, acc)

    def _validate(self):
        self.validation_batcher.cursor = 0
        inputs, labels = self.validation_batcher.get_batch()
        state = np.zeros([self.validation_batcher.batch_size, self.model.nlayers * self.model.cell_size])

        feed_dict = {self.model.X: inputs, self.model.Y_: labels, self.model.Hin: state,
                     self.model.pkeep: 1.0, self.model.batch_size: self.validation_batcher.batch_size}
        ls, acc, smm = self._sess.run([self.model.batchloss, self.model.accuracy, self.model.summaries],
                                      feed_dict=feed_dict)

        print_validation_stats(ls, acc)
        self.validation_writer.add_summary(smm, self.batcher.batch)

    def generate_text(self, prime, length=1000):
        generated_text = prime
        state = np.zeros([1, self.model.nlayers * self.model.cell_size], dtype=np.float32)
        for c in prime[:-1]:
            x = np.array([[VOCABULARY_LOOKUP[c]]])
            feed = {self.model.X: x, self.model.pkeep: 1.0, self.model.Hin: state, self.model.batch_size: 1}
            yo, state = self._sess.run([self.model.Yo, self.model.H], feed_dict=feed)

        x = np.array([self.batcher.encode_text(prime[-1])])
        for _ in range(length):
            feed = {self.model.X: x, self.model.pkeep: 1.0, self.model.Hin: state, self.model.batch_size: 1}
            yo, state = self._sess.run([self.model.Yo, self.model.H], feed_dict=feed)
            c = weighted_pick(yo, len(VOCABULARY), topn=2)
            x = np.array([[c]])  # shape [batch_size, sequence_length] with batch_size=1 and sequence_length=1
            c = VOCABULARY[c]
            generated_text += c
        return generated_text
