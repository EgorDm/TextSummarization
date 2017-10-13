import re
import pandas as pd
from datasets.constants import contractions, meta_chars
import nltk
from nltk import corpus
nltk.download('stopwords')


def clean_text(text, remove_stopwords=True):
    # Convert words to lower case
    text = text.lower()

    # Replace contractions with their longer forms
    if True:
        text = text.split()
        new_text = []
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)

    # Format words and remove unwanted characters
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text)
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)

    if remove_stopwords:
        text = text.split()
        stops = set(corpus.stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)

    return text


def count_words(count_dict, text):
    for sentence in text:
        for word in sentence.split():
            if word not in count_dict:
                count_dict[word] = 1
            else:
                count_dict[word] += 1


def create_dictionary(word_counts, embeddings, threshold=30):
    ret = {}
    value = 0
    for word, count in word_counts.items():
        if count >= threshold or word in embeddings:
            ret[word] = value
            value += 1
    for char in meta_chars:
        ret[char] = value
        value += 1
    return ret


def inverse_dictionary(dictionary):
    return {v: k for k, v in dictionary.items()}


def convert_to_ints(text, dictionary, eos=False):
    """Convert words in text to an integer.
       If word is not in dictionary, use UNK's integer.
       Total the number of words and UNKs.
       Add EOS token to the end of texts"""
    ret = []
    for sentence in text:
        sentence_ints = []
        for word in sentence.split():
            if word in dictionary:
                sentence_ints.append(dictionary[word])
            else:
                sentence_ints.append(dictionary["<UNK>"])
        if eos:
            sentence_ints.append(dictionary["<EOS>"])
        ret.append(sentence_ints)
    return ret


def create_lengths(text):
    """Create a data frame of the sentence lengths from a text"""
    lengths = []
    for sentence in text:
        lengths.append(len(sentence))
    return pd.DataFrame(lengths, columns=['counts'])


def unk_counter(sentence, dictionary):
    """Counts the number of time UNK appears in a sentence."""
    unk_count = 0
    for word in sentence:
        if word == dictionary["<UNK>"]:
            unk_count += 1
    return unk_count


def pad_sentence_batch(sentence_batch, dictionary):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [dictionary['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]