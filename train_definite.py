from batchers.CharacterBatcher import CharacterBatcher
from models.RnnModel import RNNModel
from trainers.MainTrainer import MainTrainer
import argparse


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='data/lyrics/powermetal',
                        help='Path with all the data to train on.')
    parser.add_argument('--vali_dir', type=str, default='data/lyrics/dragonforce_validate',
                        help='Path with all the data to use for validation.')
    parser.add_argument('--log_dir', type=str, default='data/log', help='Directory to store logs in')
    parser.add_argument('--save_dir', type=str, default='data/save', help='Directory to save checkpoints in.')
    parser.add_argument('--cell_size', type=int, default=512, help='Size of cell\'s hidden state.')
    parser.add_argument('--nlayers', type=int, default=3, help='Number of cell layers.')

    pass


def run(args):
    log_path = 'data/log'
    save_path = 'data/save'

    # batch_size = 60
    # sequence_length = 200
    batch_size = 30
    sequence_length = 100

    nlayers = 3
    cell_size = 512

    batcher = CharacterBatcher('data/lyrics/powermetal', False, batch_size, sequence_length)
    validation_batcher = CharacterBatcher('data/lyrics/dragonforce_validate', False, 8, 90)

    model = RNNModel(batcher, nlayers, cell_size)

    trainer = MainTrainer(batcher, validation_batcher, model, 'powermetal', log_path, save_path)

    trainer.train(10)


if __name__ == '__main__':
    main()
