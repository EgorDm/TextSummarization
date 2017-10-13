import numpy as np

def log_sample(input_seq, prediction_seq, vocabulary):
    pad = vocabulary["<PAD>"]

    print('\nText')
    print('  Word Ids:    {}'.format([i for i in input_seq]))
    print('  Input Words: {}'.format(" ".join([vocabulary[i] for i in input_seq])))

    print('\nSummary')
    print('  Word Ids:       {}'.format([i for i in prediction_seq if i != pad]))
    print('  Response Words: {}'.format(" ".join([vocabulary[i] for i in prediction_seq if i != pad])))