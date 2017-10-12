import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from tensorflow.contrib import seq2seq
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors


class Seq2SeqModel:
    def __init__(self, embedding_matrix, batch_size, vocab_length, vocab_dictionary, cell_size=512, num_layers=2,
                 kprob=0.8) -> None:
        # Constants
        self.cell_size = cell_size
        self.num_layers = num_layers
        self.kprob = kprob
        self.batch_size = batch_size
        self.vocab_length = vocab_length
        self.vocab_dictionary = vocab_dictionary

        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.input_data = tf.placeholder(tf.int32, [None, None], name='input')
        self.targets = tf.placeholder(tf.int32, [None, None], name='targets')

        self.in_length = tf.placeholder(tf.int32, (None,), name='in_length')
        self.out_length = tf.placeholder(tf.int32, (None,), name='out_length')
        self.out_max_length = tf.reduce_max(self.out_length, name='out_max_length')

        # Embeddings
        self.embeddings = embedding_matrix

        # Encoder
        encoder_embed_input = tf.nn.embedding_lookup(self.embeddings, tf.reverse(self.input_data, [-1]))
        encoder_output, encode_state = self.encoding_layer(encoder_embed_input)

        # Decoder
        decoder_input = self.process_encoding_input(self.targets, self.batch_size)
        decoder_embed_input = tf.nn.embedding_lookup(self.embeddings, decoder_input)

        self.train_logits, self.inference_logits = self.decoding_layer(decoder_embed_input, encoder_output, encode_state)

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

    def encoding_layer(self, input):
        for i in range(self.num_layers):
            with tf.variable_scope('encoder_{}'.format(i)):
                cell_fw = rnn.LSTMCell(self.cell_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
                cell_fw = rnn.DropoutWrapper(cell_fw, input_keep_prob=self.kprob)

                cell_bw = rnn.LSTMCell(self.cell_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
                cell_bw = rnn.DropoutWrapper(cell_bw, input_keep_prob=self.kprob)

                output, state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input, self.in_length,
                                                                dtype=tf.float32)

        output = tf.concat(output, 2)
        return output, state

    def decoding_layer(self, input, encoder_output, encoder_state):
        for i in range(self.num_layers):
            with tf.variable_scope('decoder_{}'.format(i)):
                decoder_cell = rnn.LSTMCell(self.cell_size,
                                            initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
                decoder_cell = rnn.DropoutWrapper(decoder_cell, input_keep_prob=self.keep_prob)

        output_layer = Dense(self.vocab_length,
                             kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        attention_mech = seq2seq.BahdanauAttention(self.cell_size, encoder_output, self.in_length, normalize=False)
        decoder_cell = seq2seq.DynamicAttentionWrapper(decoder_cell, attention_mech, self.cell_size)

        zero_state = _zero_state_tensors(self.cell_size, self.batch_size, tf.float32)
        initial_state = seq2seq.DynamicAttentionWrapperState(encoder_state[0], zero_state)

        with tf.variable_scope("decode"):
            train_logits = self.train_decoding_layer(input, decoder_cell, initial_state, output_layer)

        with tf.variable_scope("decode", reuse=True):
            inference_logits = self.inference_decoding_layer(self.embeddings, decoder_cell, initial_state, output_layer)

        return train_logits, inference_logits

    def train_decoding_layer(self, inputs, decoder_cell, initial_state, output_layer):
        helper = seq2seq.TrainingHelper(inputs, self.out_length, time_major=False)
        decoder = seq2seq.BasicDecoder(decoder_cell, helper, initial_state, output_layer)

        logits, _ = seq2seq.dynamic_decode(decoder, output_time_major=False, impute_finished=True,
                                           maximum_iterations=self.out_max_length)

        return logits

    def inference_decoding_layer(self, embeddings, decoder_cell, initial_state, output_layer):
        start_token = self.vocab_dictionary['<GO>']
        end_token = self.vocab_dictionary['<EOS>']
        start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [self.batch_size], name='start_tokens')

        helper = seq2seq.GreedyEmbeddingHelper(embeddings, start_tokens, end_token)
        decoder = seq2seq.BasicDecoder(decoder_cell, helper, initial_state, output_layer)

        logits, _ = seq2seq.dynamic_decode(decoder, False, True,maximum_iterations=self.out_max_length)
        return logits