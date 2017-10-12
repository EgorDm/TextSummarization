import time
import datasets.NewsSummaryDataset as ds
from batchers.Batcher import Batcher
import tensorflow as tf
from tqdm import tqdm, trange
from models.Seq2SeqModel import Seq2SeqModel

batch_size = 16
cell_size = 256
num_layers = 2
learning_rate = 0.005
keep_probability = 0.75

inputs, targets = ds.get_data()
batcher = Batcher(inputs, targets, batch_size)
model = Seq2SeqModel(batcher, cell_size, num_layers)

learning_rate_decay = 0.95
min_learning_rate = 0.0005
display_step = 10
pad = batcher.vocab_id["<PAD>"]

step = 0

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

while True:
    t = trange(display_step)
    for i in t:
        inputs, targets, input_lengths, target_lengths = batcher.get_batch()
        feed_dict = {
            model.inputs: inputs,
            model.targets: targets,
            model.lr: learning_rate,
            model.in_length: input_lengths,
            model.out_length: target_lengths,
            model.keep_prob: keep_probability
        }
        _, loss = sess.run([model.optimizer, model.loss], feed_dict)

        t.set_postfix(step=step, loss=loss)
        step += 1

    if step % (display_step * 3):
        inputs, targets, input_lengths, target_lengths = batcher.get_batch()
        feed_dict = {
            model.inputs: inputs,
            model.targets: targets,
            model.lr: 0,
            model.in_length: input_lengths,
            model.out_length: target_lengths,
            model.keep_prob: 1
        }
        logits = sess.run([model.inference_logits], feed_dict)

        print('\nText')
        print('  Word Ids:    {}'.format([i for i in inputs]))
        print('  Input Words: {}'.format(" ".join([batcher.id_vocab[i] for i in inputs])))

        print('\nSummary')
        print('  Word Ids:       {}'.format([i for i in logits if i != pad]))
        print('  Response Words: {}'.format(" ".join([batcher.id_vocab[i] for i in logits if i != pad])))


