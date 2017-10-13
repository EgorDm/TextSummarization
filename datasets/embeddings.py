import numpy as np

_embeddings = None


def get_embeddings():
    global _embeddings
    if _embeddings is not None: return _embeddings

    _embeddings = {}
    with open('data/numberbatch-en-1706/numberbatch-en-17.06.txt', encoding='utf-8') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            _embeddings[word] = embedding
    return _embeddings


def get_embedding_matrix(dictionary, dims=300):
    nb_words = len(dictionary)
    ret = np.zeros([nb_words, dims], dtype=np.float32)
    embeddings = get_embeddings()
    for word, i in dictionary.items():
        if word in embeddings:
            ret[i] = embeddings[word]
        else:  # If word is not in embeddings, add it
            new_embedding = np.array(np.random.uniform(-1.0, 1.0, dims))
            embeddings[word] = new_embedding
            ret[i] = new_embedding
    return ret
