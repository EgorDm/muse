from batchers.CharacterBatcher import CharacterBatcher
from models.RnnModel import RNNModel
from trainers.MainTrainer import MainTrainer
from utils import utils, config
from utils.Logger import print_generated_text
import argparse


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, help='Config you want to load defaults from')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint you want to use')
    parser.add_argument('--prime', type=str, default='The ', help='Text to use to generate more text')
    parser.add_argument('--length', type=int, default=1000, help='Length of the generated text')
    settings = parser.parse_args()
    run(settings)


def run(settings):
    myconfig = config.load_config(settings.config)
    if myconfig is None: return

    batcher = CharacterBatcher([], myconfig.lc, False, myconfig.batch_size,
                               myconfig.seq_length)
    model = RNNModel(batcher, myconfig.nlayers, myconfig.cell_size, cell_type=utils.get_cell_type(myconfig.cell))
    trainer = MainTrainer(batcher, None, model, myconfig)
    text = trainer.generate_text(settings.prime, settings.length)
    print_generated_text(text)


if __name__ == '__main__':
    main()
