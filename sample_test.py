from batchers.CharacterBatcher import CharacterBatcher
from models.KerasEmbeddingModel import KerasEmbeddingModel
from trainers.KerasTrainer import KerasTrainer
from utils import utils, config
from utils.Logger import print_generated_text
import argparse
import math, time
import tensorflow as tf


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, required=True, help='Config you want to load defaults from')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint you want to use')
    parser.add_argument('--save', default='last_model', type=str, help='Name of file to save text to.')
    parser.add_argument('--save_dir', default='data/samples', type=str, help='Name of file to save text to.')
    parser.add_argument('--prime', type=str, default='The ', help='Text to use to generate more text')
    parser.add_argument('--length', type=int, default=1000, help='Length of the generated text')
    settings = parser.parse_args()
    run(settings)


def run(settings):
    myconfig = config.load_config(settings.config)
    if myconfig is None: return
    if myconfig.lc:
        settings.prime = settings.prime.lower()
    timestamp = str(math.trunc(time.time()))

    batcher = CharacterBatcher([], myconfig.lc, False, myconfig.batch_size,
                               myconfig.seq_length)
    model = KerasEmbeddingModel(batcher, myconfig.nlayers, myconfig.cell_size, 1.0- myconfig.kprob, myconfig.lr)

    trainer = KerasTrainer(batcher, None, model, myconfig)

    with tf.Session() as sess:
        trainer._sess = sess
        trainer.model.model.load_weights('{}/{}.h5'.format(myconfig.save_dir, settings.checkpoint))
        text = trainer.generate_text(settings.prime, settings.length)
        print_generated_text(text)

    utils.mkdir_or_not(settings.save_dir)
    save_path = '{}/{}_{}.txt'.format(settings.save_dir, settings.save, timestamp)
    with open(save_path, 'w') as f:
        f.write(text)


if __name__ == '__main__':
    main()
