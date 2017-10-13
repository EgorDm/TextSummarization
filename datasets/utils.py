import os
import pickle


def save_data(data, path):
    with open(path, 'wb') as file:
        pickle.dump(data, file)


def load_data(path):
    if not os.path.exists(path):
        raise Exception('File has not been found')
    with open(path, 'rb') as file:
        return pickle.load(file)
