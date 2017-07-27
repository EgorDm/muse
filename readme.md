# Muse  - text generation with style
Generating text or lyrics with deep learning.
This project was created for testing purposes.

## Demo
Check out ```data/samples``` directory to view a few generated samples. These samples are generating by training on a few popular powermetal bands.

A few short fragments:
```
arrow to the knees
when you're the child of a battle in the darkness
i've been there before
the call of the darkness to the end of a the time
the time has come
breaking the call
the cries, of only day
```
```
thou shall not be there to see
all the pain inside
metal is for everyone.
you can't see my love is a cries
i hear the cruel of metal
i am a master of my dragons
when it's a play you will not fail to look in your hands
when it can have you see another day
another time, only tries
```

## Usage
### Download data
This project is mostly targeted towards generating lyrics. Therefore you can use ```grab_lyrics.py``` to download song lyrics by artist from websites: darklyrics and azlyrics.

Run the script
```./grab_lyrics.py```

Fill in the name of the artist without spaces and in lowercase. Use ```,``` to separate multiple artists.

Pick a source darklyrics or azlyrics

All the data will be placed inside ```data/lyrics/artists_name``` 

### Training
```./train.py --data_dir="data/lyrics/artists_name" --vali_dir="data/lyrics/artists_name" --name="trainingnr1"```

You can use multiple directories by separating them with ;

You can specify multiple different arguments:
```
--data_dir DATA_DIR   Path with all the data to train on. Separate them with
                        ; (default: data/lyrics/dragonforce)
--vali_dir VALI_DIR   Path with all the data to use for validation.
                    (default: data/lyrics/dragonforce_validate)
--log_dir LOG_DIR     Directory to store logs in (default: data/log)
--save_dir SAVE_DIR   Directory to save checkpoints in. (default: data/save)
--name NAME           Name of the session. Will be used to identify saves
                    and logs. (default: mymodel)
--cell_size CELL_SIZE
                    Size of cell's hidden state. (default: 512)
--nlayers NLAYERS     Number of cell layers. (default: 3)
--cell CELL           Cell type [lstm, gru] (default: gru)
--batch_size BATCH_SIZE
                    Size of a batch (default: 50)
--seq_length SEQ_LENGTH
                    Length of a sequence (default: 100)
--num_epochs NUM_EPOCHS
                    Number of epochs to loop (default: 50)
--display_freq DISPLAY_FREQ
                    Display log frequency (default: 50)
--lr LR               Learning rate (default: 0.001)
--kprob KPROB         Keep probability in the dropout layer (default: 0.8)
--prime PRIME         Text to use to generate more text (default: The )
--lc LC               Lowercase all the training data. (default: True)
--model MODEL         Model you want to use for training. Currently you can
                        choose between (embedding, onehot) (default:
                        embedding)

```

### Generate sample
```
./sample.py --config="powermetal_1501175906" --checkpoint="powermetal_1501175906-8000" --name="powermetal" --prime="Arrow to the knee "
```

Use saved config and checkpoint to generate text.

You can specify multiple different arguments:
```
--config CONFIG       Config you want to load defaults from (default: None)
--checkpoint CHECKPOINT
                    Checkpoint you want to use (default: None)
--save SAVE           Name of file to save text to. (default: last_model)
--save_dir SAVE_DIR   Name of file to save text to. (default: data/samples)
--prime PRIME         Text to use to generate more text (default: The )
--length LENGTH       Length of the generated text (default: 1000)
```

## Models
Check out the ```models``` folder to see which models are available.

Embedding model - embeds all the characters and trains a rnn on that

OneHot model - one hot encodes all the characters and trains a rnn on that
