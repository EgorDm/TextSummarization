import datasets.NewsSummaryDataset as ds
from batchers.Batcher import Batcher
import tensorflow as tf
from tqdm import tqdm, trange
from models.Seq2SeqModel import Seq2SeqModel
import numpy as np
import os
from utils.log_utils import log_sample

model_name = 'testboi'

batch_size = 6
cell_size = 256
num_layers = 2
keep_probability = 0.75
learning_rate = 0.005
decay_rate = 0.90
decay_step = 600

display_step = 100
display_sample = 400
save_freq = display_step * 6

save_path = 'saves/{}'.format(model_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)

inputs, targets = ds.get_data() #TODO: move to batcher a fn
batcher = Batcher(inputs, targets, batch_size, save_path)
model = Seq2SeqModel(batcher, cell_size, num_layers)

decayed_learning_rate = learning_rate
step = 0

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()

while True:
    t = trange(display_step)
    for i in t:
        inputs, targets, input_lengths, target_lengths = batcher.get_batch()
        feed_dict = {
            model.inputs: inputs,
            model.targets: targets,
            model.lr: decayed_learning_rate,
            model.in_length: input_lengths,
            model.out_length: target_lengths,
            model.keep_prob: keep_probability
        }
        _, loss = sess.run([model.optimizer, model.loss], feed_dict)

        t.set_postfix(step=step, loss=loss, lr=decayed_learning_rate, cursor=batcher.cursor)
        step += 1

    if step % display_sample == 0:
        inputs, targets, input_lengths, target_lengths = batcher.get_batch()
        text = inputs[0]
        feed_dict = {
            model.inputs: [text] * batch_size,
            model.in_length: [len(text)] * batch_size,
            model.out_length: [np.random.randint(30, 50)],
            model.lr: 0,
            model.keep_prob: 1
        }
        logits = sess.run([model.inference_logits], feed_dict)
        log_sample(text, logits[0][0], batcher.vocab_id, batcher.id_vocab)

    if step % save_freq == 0:
        save_path = saver.save(sess, '{}/{}.ckpt'.format(save_path, model_name), global_step=step)
        print('Saved checkpoint to {}'.format(save_path))

    decayed_learning_rate = learning_rate * pow(decay_rate, (step / decay_rate))
