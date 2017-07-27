from models.BaseModel import *
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers


class OneHotModel(BaseModel):
    def __init__(self, batcher, nlayers: int, cell_size: int, cell_type=rnn.GRUCell):
        super().__init__()
        # Models parameters
        self.nlayers = nlayers
        self.cell_size = cell_size

        # Hyper parameters
        self.lr = tf.placeholder(tf.float32, name='lr')
        self.pkeep = tf.placeholder(tf.float32, name='pkeep')
        self.batch_size = tf.placeholder(tf.int32, name='batch_size')

        # Inputs
        self.X = tf.placeholder(tf.uint8, [None, None], name='X')
        # X: [batch_size, sequence_length]
        Xo = tf.one_hot(self.X, batcher.get_vocabulary_size(), 1.0, 0.0)
        # Xo: [batch_size, sequence_length, vocabulary_length]

        # Labels = inputs shifted by one character
        self.Y_ = tf.placeholder(tf.uint8, [None, None], name='Y_')
        # Y_: [batch_size, sequence_length]
        Yo_ = tf.one_hot(self.Y_, batcher.get_vocabulary_size(), 1.0, 0.0)
        # Yo_: [batch_size, sequence_length]

        # Cell state
        self.Hin = tf.placeholder(tf.float32, [None, cell_size * nlayers], name='Hin')
        # Hin: [batch_size, cell_size * n_layers]

        # Cells
        multicell = create_multicell(nlayers, cell_type, cell_size, self.pkeep)

        # RNN
        Yr, H = tf.nn.dynamic_rnn(multicell, Xo, dtype=tf.float32, initial_state=self.Hin)
        # Yr: [ batch_size, sequence_length, cell_size ]
        # H:  [ batch_size, cell_size * n_layers ] last state

        # Give final state a name
        self.H = tf.identity(H, name='H')

        # Outputs
        Yflat = tf.reshape(Yr, [-1, cell_size])
        # Yflat: [batch_size * sequence_length, cell_size]
        Ylogits = layers.linear(Yflat, batcher.get_vocabulary_size())
        # Ylogits: [batch_size * sequence_length, vocabulary_length]

        # Labels
        Yflat_ = tf.reshape(Yo_, [-1, batcher.get_vocabulary_size()])
        # Yflat_ [batch_size * sequence_length, vocabulary_length]

        # Loss
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Yflat_)
        # loss: [ batch_size x sequence_length]
        loss = tf.reshape(loss, [self.batch_size, -1])
        # loss: [batch_size, sequence_length]

        # Predictions
        self.Yo = tf.nn.softmax(Ylogits, name='Yo')
        # Yo: [batch_size * sequence_length, vocabulary_length]
        Y = tf.argmax(self.Yo, 1)
        # Y: [batch_size * sequence_length]
        self.Y = tf.reshape(Y, [self.batch_size, -1], name="Y")
        # Y: [batch_size, sequence_length]

        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(loss)

        # Stats
        self.seqloss = tf.reduce_mean(loss, 1)
        self.batchloss = tf.reduce_mean(self.seqloss)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.Y_, tf.cast(self.Y, tf.uint8)), tf.float32))
        loss_summary = tf.summary.scalar("batch_loss", self.batchloss)
        acc_summary = tf.summary.scalar("batch_accuracy", self.accuracy)
        self.summaries = tf.summary.merge([loss_summary, acc_summary])
