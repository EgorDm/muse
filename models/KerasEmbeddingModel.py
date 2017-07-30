from keras.engine import InputLayer
from keras.layers import Embedding, GRU, Dense, Activation
from keras.models import Sequential
from keras.optimizers import Adam


class KerasEmbeddingModel:
    def __init__(self, batcher, nlayers: int, cell_size: int, dropout: float, lr: float):
        super().__init__()
        # Models parameters
        self.nlayers = nlayers
        self.cell_size = cell_size

        model = Sequential()
        model.add(InputLayer(batch_input_shape=(None, None)))
        # [batch_size, sequence_length]

        model.add(Embedding(batcher.get_vocabulary_size(), self.cell_size))
        # input_embeddings: [batch_size, sequence_length, embeddings_dimensions]

        model.add(GRU(cell_size, return_sequences=True, dropout=dropout))
        for _ in range(nlayers - 1):
            model.add(GRU(cell_size, dropout=dropout, return_sequences=True))
        # [ batch_size, sequence_length, cell_size ]

        model.add(Dense(batcher.get_vocabulary_size()))
        # [batch_size, sequence_length, vocabulary_length]

        model.add(Activation('softmax'))

        optimizer = Adam(lr=lr)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        self.model = model
