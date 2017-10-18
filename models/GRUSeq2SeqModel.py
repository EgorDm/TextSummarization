import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from tensorflow.contrib import seq2seq


def build_multicell(nlayers=2, cell_size=128, kprob=0.8):
    return rnn.MultiRNNCell([
        rnn.DropoutWrapper(rnn.GRUCell(cell_size), input_keep_prob=kprob) for _ in range(nlayers)
    ], state_is_tuple=True)


class GRUSeq2SeqModel:
    def __init__(self, batcher, cell_size=256, bidir_layers=2, uni_layers=2):
        # Constants
        self.uni_layers = uni_layers
        self.bidir_layers = bidir_layers
        self.cell_size = cell_size
        self.batch_size = batcher.batch_size
        self.vocab_length = len(batcher.vocab_id)
        self.vocab_dictionary = batcher.vocab_id

        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.inputs = tf.placeholder(tf.int32, [None, None], name='input')
        self.targets = tf.placeholder(tf.int32, [None, None], name='targets')

        self.in_length = tf.placeholder(tf.int32, (None,), name='in_length')
        self.out_length = tf.placeholder(tf.int32, (None,), name='out_length')
        self.out_max_length = tf.reduce_max(self.out_length, name='out_max_length')

        # Embeddings
        self.embeddings = batcher.embeddings

        # Encoder
        encoder_embed_input = tf.nn.embedding_lookup(self.embeddings, tf.reverse(self.inputs, [-1]))
        encoder_output, encode_state = self.encoding_layer(encoder_embed_input)

        # Decoder
        decoder_input = self.process_encoding_input(self.targets, self.batch_size)
        decoder_embed_input = tf.nn.embedding_lookup(self.embeddings, decoder_input)

        self.train_logits, self.inference_logits = self.decoding_layer(decoder_embed_input, encoder_output,
                                                                       encode_state)

        self.train_logits = tf.identity(self.train_logits.rnn_output, 'logits')
        self.inference_logits = tf.identity(self.inference_logits.sample_id, name='predictions')

        # Create the weights for sequence_loss
        masks = tf.sequence_mask(self.out_length, self.out_max_length, dtype=tf.float32, name='masks')

        with tf.name_scope("optimization"):
            # Loss function
            self.loss = tf.contrib.seq2seq.sequence_loss(self.train_logits, self.targets, masks)

            # Optimizer
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def process_encoding_input(self, target_data, batch_size):
        """
        Prepend starting word (<GO>) to every sequence
        :param target_data:
        :param batch_size:
        :return:
        """
        ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
        dec_input = tf.concat([tf.fill([batch_size, 1], self.vocab_dictionary['<GO>']), ending], 1)
        return dec_input

    def encoding_layer(self, rnn_inputs):
        cell_fw = build_multicell(self.bidir_layers, self.cell_size, self.keep_prob)
        cell_bw = build_multicell(self.bidir_layers, self.cell_size, self.keep_prob)

        output, state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, rnn_inputs, self.in_length,
                                                        dtype=tf.float32)
        output = tf.concat(output, 2)
        return output, state

    def decoding_layer(self, rnn_inputs, encoder_output, encoder_state):
        decoder_cell = build_multicell(self.uni_layers, self.cell_size, self.keep_prob)

        attention_mech = seq2seq.BahdanauAttention(self.cell_size, encoder_output, self.in_length)
        attention_cell = seq2seq.AttentionWrapper(decoder_cell, attention_mech, self.cell_size / 2)
        decoder_cell = rnn.OutputProjectionWrapper(attention_cell, self.vocab_length)

        initial_state = decoder_cell.zero_state(self.batch_size, tf.float32)
        initial_state.clone(cell_state=encoder_state)

        with tf.variable_scope("decode"):
            train_logits = self.train_decoding_layer(rnn_inputs, decoder_cell, initial_state)

        with tf.variable_scope("decode", reuse=True):
            inference_logits = self.inference_decoding_layer(self.embeddings, decoder_cell, initial_state)

        return train_logits, inference_logits

    def train_decoding_layer(self, inputs, decoder_cell, initial_state):
        helper = seq2seq.TrainingHelper(inputs, self.out_length, time_major=False)
        decoder = seq2seq.BasicDecoder(decoder_cell, helper, initial_state)

        outputs = seq2seq.dynamic_decode(decoder, output_time_major=False, impute_finished=True,
                                         maximum_iterations=self.out_max_length)
        return outputs[0]

    def inference_decoding_layer(self, embeddings, decoder_cell, initial_state):
        start_token = self.vocab_dictionary['<GO>']
        end_token = self.vocab_dictionary['<EOS>']
        start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [self.batch_size], name='start_tokens')

        helper = seq2seq.GreedyEmbeddingHelper(embeddings, start_tokens, end_token)
        decoder = seq2seq.BasicDecoder(decoder_cell, helper, initial_state)

        outputs = seq2seq.dynamic_decode(decoder, output_time_major=False, impute_finished=True,
                                         maximum_iterations=self.out_max_length)
        return outputs[0]
