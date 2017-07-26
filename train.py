from batchers.CharacterBatcher import CharacterBatcher
from models.RnnModel import RNNModel
from models.EmbeddingModel import EmbeddingModel
from trainers.MainTrainer import MainTrainer
import argparse
from utils import utils, config



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
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--kprob', type=float, default=0.8, help='Keep probability in the dropout layer')
    parser.add_argument('--prime', type=str, default='The ', help='Text to use to generate more text')
    parser.add_argument('--lc', type=bool, default=True, help='Lowercase all the training data.')
    settings = parser.parse_args()
    run(settings)


def run(settings):
    if settings.lc:
        settings.prime = settings.prime.lower()
    batcher = CharacterBatcher(settings.data_dir.split(';'), settings.lc, False, settings.batch_size,
                               settings.seq_length)
    validation_batcher = CharacterBatcher([settings.vali_dir], settings.lc, False, 8, 90)

    # model = RNNModel(batcher, settings.nlayers, settings.cell_size, cell_type=utils.get_cell_type(settings.cell))
    model = EmbeddingModel(batcher, settings.nlayers, settings.cell_size, cell_type=utils.get_cell_type(settings.cell))

    trainer = MainTrainer(batcher, validation_batcher, model, settings)

    config.save_config(settings, trainer.model_name)
    trainer.train(settings.num_epochs)


if __name__ == '__main__':
    main()
