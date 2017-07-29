from trainers.BaseTrainer import BaseTrainer
from utils.Logger import print_learning_learned_comparison, Progress
import numpy as np

from utils.utils import hot_one


class KerasTrainer(BaseTrainer):
    def __init__(self, batcher, validation_batcher, model, settings):
        super().__init__(batcher, validation_batcher, model, settings)

    def train(self, epochs: int = 10):
        progress = Progress(self.display_freq, size=111 + 2,
                            msg="Training on next " + str(self.display_freq) + " batches")

        while self.batcher.epoch <= epochs:
            self._train_step_keras()

            progress.step(reset=self.batcher.batch % self.display_freq == 0)

    def _train_step_keras(self):
        inputs, labels = self.batcher.get_batch()
        self.model.model.train_on_batch(inputs, hot_one(labels, self.batcher.get_vocabulary_size()))

        if self.batcher.batch % self.display_freq == 0:
            self._train_log_keras(inputs, hot_one(labels, self.batcher.get_vocabulary_size()))

    def _train_log_keras(self, inputs: np.ndarray, labels: np.ndarray):
        score, acc = self.model.model.evaluate(inputs, labels, batch_size=self.batcher.batch_size)
        y = self.model.model.predict(inputs, batch_size=self.batcher.batch_size)
        y = np.argmax(y, 2)
        loss = []
        for k in range(self.batcher.batch_size):
            loss.append(score)
        print_learning_learned_comparison(self.batcher, inputs, y, loss, score, acc)
