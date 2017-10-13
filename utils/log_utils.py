import numpy as np

def log_sample(input_seq, prediction_seq, vocab, vocab_lookup):
    pad = vocab["<PAD>"]

    print('\nText')
    print('  Input Ids:    {}'.format([i for i in input_seq]))
    print('  Input Words: {}'.format(" ".join([vocab_lookup[i] for i in input_seq])))

    print('\nSummary')
    print('  Response Ids:       {}'.format([i for i in prediction_seq if i != pad]))
    print('  Response Words: {}'.format(" ".join([vocab_lookup[i] for i in prediction_seq if i != pad])))