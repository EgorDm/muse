from trainers.BaseTrainer import BaseTrainer
from utils.Logger import print_learning_learned_comparison, Progress, print_validation_stats, print_generated_text
import keras.backend.tensorflow_backend as K
import numpy as np
import tensorflow as tf

from utils.utils import hot_one, weighted_pick


class KerasTrainer(BaseTrainer):
    _sess = None

    def __init__(self, batcher, validation_batcher, model, settings):
        super().__init__(batcher, validation_batcher, model, settings)

    def train(self, epochs: int = 10):
        self._sess = tf.Session()
        K.set_session(self._sess)

        progress = Progress(self.display_freq, size=111 + 2,
                            msg="Training on next " + str(self.display_freq) + " batches")

        while self.batcher.epoch <= epochs:
            self._train_step_keras()

            if self.batcher.batch % self.display_freq == 0 and self.validation_batcher is not None:
                self._validate()

            if self.batcher.batch % (self.display_freq * 5) == 0:
                text = self.generate_text(self.prime)
                print_generated_text(text)

            if self.batcher.batch % (self.display_freq * 10) == 0:
                save_path = '{}/{}_{}.h5'.format(self.save_path, self.model_name, self.batcher.batch)
                self.model.model.save_weights(save_path)
                print("Saved file: " + save_path)

            progress.step(reset=self.batcher.batch % self.display_freq == 0)

    def _train_step_keras(self):
        inputs, labels = self.batcher.get_batch()
        self.model.model.train_on_batch(inputs, hot_one(labels, self.batcher.get_vocabulary_size()))

        if self.batcher.batch % self.display_freq == 0:
            self._train_log_keras(inputs, labels)

    def _train_log_keras(self, inputs: np.ndarray, labels: np.ndarray):
        score, acc = self.model.model.evaluate(inputs, hot_one(labels, self.batcher.get_vocabulary_size()),
                                               batch_size=self.batcher.batch_size)
        y = self.model.model.predict(inputs, batch_size=self.batcher.batch_size)
        y = np.argmax(y, 2)
        loss = []
        for k in range(self.batcher.batch_size):
            loss.append(score)
        print_learning_learned_comparison(self.batcher, inputs, y, loss, score, acc)

    def _validate(self):
        self.validation_batcher.cursor = 0
        inputs, labels = self.validation_batcher.get_batch()
        score, acc = self.model.model.evaluate(inputs, hot_one(labels, self.batcher.get_vocabulary_size()),
                                               batch_size=self.batcher.batch_size)
        print_validation_stats(score, acc)

    def generate_text(self, prime, length=300):
        generated_text = prime
        if len(generated_text) < self.batcher.sequence_length:
            generated_text = ' '*(self.batcher.sequence_length-len(generated_text)) + generated_text

        x = np.array([self.batcher.encode_text(generated_text)])
        for _ in range(length):
            y = self.model.model.predict_on_batch(x)
            c = weighted_pick(y[0][-1], self.batcher.get_vocabulary_size(), topn=2)
            generated_text += self.batcher.get_vocabulary()[c]
            x[0] = np.append(x[0][1:], c)

        return generated_text
