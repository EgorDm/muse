import pickle
from utils import utils

CONFIG_PATH = 'data/config'


def save_config(config, name):
    utils.mkdir_or_not(CONFIG_PATH)
    file = '{}/{}.pickle'.format(CONFIG_PATH, name)
    pickle.dump(config, open(file, "wb"))


def load_config(name):
    file = '{}/{}.pickle'.format(CONFIG_PATH, name)
    return pickle.load(open(file, "rb"))
