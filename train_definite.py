from batchers.CharacterBatcher import CharacterBatcher
from models.RnnModel import RNNModel
from trainers.MainTrainer import MainTrainer

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
