import os
import numpy as np
import tensorflow as tf

from batchers.Batcher import Batcher
from models.Seq2SeqModel import Seq2SeqModel
from utils.log_utils import log_sample

model_name = 'testboi'
load_step = 10

# TODO: save || const
batch_size = 6
cell_size = 256
num_layers = 2

save_path = 'saves/{}'.format(model_name)
if not os.path.exists(save_path):
    print('No such model exists in {}'.format(save_path))
    exit()

batcher = Batcher(None, None, batch_size, save_path)
model = Seq2SeqModel(batcher, cell_size, num_layers)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()
saver.restore(sess, '{}/checkpoints/{}.ckpt-{}'.format(save_path, model_name, load_step))

# saver = tf.train.import_meta_graph('{}/{}.ckpt.meta'.format(save_path, model_name))
# saver.restore(sess, tf.train.latest_checkpoint(save_path))

# TODO: user input
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

sess.close()

