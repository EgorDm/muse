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
