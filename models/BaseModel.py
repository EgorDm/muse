import tensorflow as tf
from tensorflow.contrib import rnn


class BaseModel:
    def __init__(self):
        pass


def create_multicell(nlayers: int, cell_type, cell_size: int, pkeep: float):
    cells = [create_cell(cell_type, cell_size, pkeep) for _ in range(nlayers)]
    multicell = rnn.MultiRNNCell(cells, state_is_tuple=False)
    multicell = rnn.DropoutWrapper(multicell, output_keep_prob=pkeep)
    return multicell


def create_cell(cell_type, cell_size: int, pkeep: float):
    cell = cell_type(cell_size)
    return rnn.DropoutWrapper(cell, input_keep_prob=pkeep)


def get_cell_type(cell_type):
    if cell_type == 'simple_lstm':
        return tf.nn.rnn_cell.BasicLSTMCell
    elif cell_type == 'lstm':
        return tf.nn.rnn_cell.LSTMCell
    elif cell_type == 'gru':
        return tf.nn.rnn_cell.GRUCell
    elif cell_type == 'simple_classic_rnn':
        return tf.nn.rnn_cell.BasicRNNCell
    elif cell_type == 'classic_rnn':
        return tf.nn.rnn_cell.RNNCell
    else:
        raise Exception('No such cell type {}'.format(cell_type))
