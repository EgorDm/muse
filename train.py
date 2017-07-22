from batchers.CharacterBatcher import CharacterBatcher
from models.RnnModel import RNNModel
from trainers.MainTrainer import MainTrainer
import argparse
import tensorflow as tf


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='data/lyrics/dragonforce',
                        help='Path with all the data to train on. Separate them with ;')
    parser.add_argument('--vali_dir', type=str, default='data/lyrics/dragonforce_validate',
                        help='Path with all the data to use for validation.')
    parser.add_argument('--log_dir', type=str, default='data/log', help='Directory to store logs in')
    parser.add_argument('--save_dir', type=str, default='data/save', help='Directory to save checkpoints in.')
    parser.add_argument('--name', type=str, default='mymodel',
                        help='Name of the session. Will be used to identify saves and logs.')
    parser.add_argument('--cell_size', type=int, default=512, help='Size of cell\'s hidden state.')
    parser.add_argument('--nlayers', type=int, default=3, help='Number of cell layers.')
    parser.add_argument('--cell', type=str, default='gru', help='Cell type [lstm, gru]')
    parser.add_argument('--batch_size', type=int, default=50, help='Size of a batch')
    parser.add_argument('--seq_length', type=int, default=100, help='Length of a sequence')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to loop')
    parser.add_argument('--display_freq', type=int, default=50, help='Display log frequency')
    parser.add_argument('--lr', type=float, default=0.002, help='Learning rate')
    parser.add_argument('--kprob', type=float, default=0.8, help='Keep probability in the dropout layer')
    parser.add_argument('--prime', type=str, default='The ', help='Text to use to generate more text')
    settings = parser.parse_args()
    run(settings)


def run(settings):
    batcher = CharacterBatcher(settings.data_dir.split(';'), False, settings.batch_size, settings.seq_length)
    validation_batcher = CharacterBatcher([settings.vali_dir], False, 8, 90)

    model = RNNModel(batcher, settings.nlayers, settings.cell_size, cell_type=get_cell_type(settings.cell))

    trainer = MainTrainer(batcher, validation_batcher, model, settings)

    trainer.train(10)


def get_cell_type(cell_type):
    if cell_type == 'lstm':
        return tf.nn.rnn_cell.LSTMCell
    elif cell_type == 'gru':
        return tf.nn.rnn_cell.GRUCell
    # elif cell_type == 'simple_classic_rnn':
    #     return tf.nn.rnn_cell.BasicRNNCell
    # elif cell_type == 'classic_rnn':
    #     return tf.nn.rnn_cell.RNNCell
    else:
        raise Exception('No such cell type {}'.format(cell_type))

if __name__ == '__main__':
    main()
