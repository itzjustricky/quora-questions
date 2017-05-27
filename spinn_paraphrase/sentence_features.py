"""
    Calculate some specific features for the
    sentences that may help the architecture

    The extra features to be added are:

    Plain Features
    ===============
    1. Difference in sentence length
    2. Percentage of words and phrases in one sentence that are in other sentence and vice-versa
      a. sentence 1
      b. sentence 2

    Number Features
    ===============
    1. If the two sentences contain exactly the same numbers or no number
    2. If two sentences contain the same numbers
    3. If the set of numbers of one sentence is the strict subset of the numbers in the other sentence

"""

import nltk

import numpy as np

import torch
from torch.autograd import Variable


def sentence_features(sentence1, sentence2, sentence_vocab, use_gpu=True):
    """
    :type sentence1: torch.Variable
    :type sentence2: torch.Variable
    :param sentence1: a batch of first set of sentences (in numerical form)
    :param sentence2: a batch of second set of sentences (in numerical form)
    :param sentence_vocab: the vocabulary used to translate the numerical forms
        of the sentences into the word form
    """
    # sentence1 = sentence1.data.numpy()
    sentence1_np = sentence1.data.cpu().numpy()
    sentence2_np = sentence2.data.cpu().numpy()

    batched_sentence_features = []
    for sent1, sent2 in zip(sentence1_np.T, sentence2_np.T):
        word_sent1 = translate_numerical_sentences(sent1, sentence_vocab)
        word_sent1 = [word for word in word_sent1 if word != '<pad>']
        word_sent2 = translate_numerical_sentences(sent2, sentence_vocab)
        word_sent2 = [word for word in word_sent2 if word != '<pad>']
        batched_sentence_features.append(sentence_features_for_one_sample(word_sent1, word_sent2))

    # (7xbatch_size) tensor
    feature_tensor = convert_features_to_torch(batched_sentence_features)
    if use_gpu:
        return Variable(feature_tensor.cuda())
    else:
        return Variable(feature_tensor)


def translate_numerical_sentences(sentence, sentence_vocab):
    return [sentence_vocab.itos[word] for word in sentence]


def convert_features_to_torch(batched_features):
    batched_features_np = np.array(batched_features, dtype=np.float32)
    return torch.from_numpy(batched_features_np)


def sentence_features_for_one_sample(sentence1, sentence2):
    """ Create sentence features from sentences represented by list of strings
        e.g. sentence1 = ['The', 'world', 'is', 'big']
    """
    word_features = WordFeatures(sentence1, sentence2)
    number_features = NumberFeatures(sentence1, sentence2)

    gathered_features = [
        word_features.has_same_words(),
        word_features.length_difference(),
        word_features.shared_words_percentage(1),       # % of words in sentence1 in sentence2
        word_features.shared_words_percentage(2),       # % of words in sentence2 in sentence1
        number_features.exactly_same_numbers_or_none(),
        number_features.exactly_same_numbers(),
        number_features.one_is_strict_subset(),
    ]

    return gathered_features


class WordFeatures(object):

    def __init__(self, sentence1, sentence2):
        self.sentence1 = sentence1
        self.sentence2 = sentence2
        self.sentence_set1 = set(sentence1)
        self.sentence_set2 = set(sentence2)
        self.shared_words = set.intersection(self.sentence_set1, self.sentence_set1)

    def shared_words_percentage(self, sentence_number):
        if sentence_number == 1:        # % of words in sentence 1 shared
            return self.n_words_in_set(self.sentence1, self.shared_words) / len(self.sentence1)
        elif sentence_number == 2:
            return self.n_words_in_set(self.sentence2, self.shared_words) / len(self.sentence2)
        else:
            raise ValueError("Passed an invalid sentence_number")

    def has_same_words(self):
        return (self.sentence_set1 == self.sentence_set2)

    def length_difference(self):
        return abs(len(self.sentence1) - len(self.sentence2))

    @staticmethod
    def n_words_in_set(sentence, word_set):
        cnt = 0
        for word in sentence:
            if word in word_set:
                cnt += 1
        return cnt


class NumberFeatures(object):

    def __init__(self, sentence1, sentence2):
        self.numbers_in_1 = self.numbers_in_sentence(sentence1)
        self.numbers_in_2 = self.numbers_in_sentence(sentence2)

    def exactly_same_numbers_or_none(self):
        return (self.numbers_in_1 == self.numbers_in_2) or \
               ((len(self.numbers_in_1) == 0) and (len(self.numbers_in_2) == 0))

    def exactly_same_numbers(self):
        return (set(self.numbers_in_1) == set(self.numbers_in_2))

    def one_is_strict_subset(self):
        numbers_set_1 = set(self.numbers_in_1)
        numbers_set_2 = set(self.numbers_in_2)
        return (numbers_set_1.issubset(numbers_set_2) or numbers_set_2.issubset(numbers_set_1)) and \
               (numbers_set_1 != numbers_set_2)

    @staticmethod
    def numbers_in_sentence(sentence):
        """ Returns the numbers found in a sentence

        :param sentence: list of words
        """
        sentence_pos_tags = nltk.pos_tag(sentence)

        numbers_in_sentence = []
        for word, pos_tag in sentence_pos_tags:
            if pos_tag == 'CD':
                numbers_in_sentence.append(word)
        return numbers_in_sentence
