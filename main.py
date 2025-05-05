import numpy as np
import tensorflow as tf
import os, re

file_path = tf.keras.utils.get_file('shakespeare.txt',
    'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')


with open (file_path, 'r', encoding='utf-8') as f:
    text = f.read().lower()

text = re.sub(r'[^a-z \n]', '', text)
print('text', len(text))


vocab = sorted(set(text))
char2idx = {u:i for i, u in enumerate(vocab)}
idxchar = np.array(vocab)

#endcodeing text../
text_as_int = np.array([char2idx[c] for c in text])

seq_length = 100
examples_per_epoch = len(text)  // seq_length

char_dataset = tf.data.Dataset.from_tensor_slices(text_ass_int)
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

#function for input -target seqq 
def split_input_target(chunk):
    input_seq = chunk[:-1]
    target_seq = [1:]
    return input_seq, target_seq

dataset = sequences.map(split_input_target)

BATCH_SIZE = 64
BUFFER_SIZE =10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_reminder=True)

