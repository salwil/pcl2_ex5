#!/usr/bin/env python3
#-*- coding: utf-8 -*-
# Roger Rüttimann rroger 02-914-471
# Salome Wildermuth salomew 10-289-544

from collections import defaultdict
from typing import Dict, List
from nltk.util import ngrams
import random

class NGramModel:

    def __init__(self, path_to_document: str, nr_of_lines: int, n_gram_size: int, inf_loop=False):
        """
        Class NGramModel reads a file and generates on base of the probability of n-gram sequences new songtext lines.
        :param path_to_document:    Path to input file, str
        :param nr_of_lines:         number of songtext lines to be generated, int
        :param n_gram_size:         size of n_grams on base of which we calculate the probabilites for follower tokens,
                                    int
        :param inf_loop:            True, if we don't want to stop generating songtext lines anymore; else False, bool
        """
        self.input = self.extract_input_file(path_to_document, n_gram_size)
        self._ngram_freqs = self.get_ngram_freqs(self.input, n_gram_size)
        if inf_loop:
            while inf_loop:
                print(' '.join(self.generate_sentence(self._ngram_freqs, n_gram_size)))
        else:
            for i in range(0, nr_of_lines):
                print(' '.join(self.generate_sentence(self._ngram_freqs, n_gram_size)))

    def extract_input_file(self, path_to_document: str, n_gram_size: int):
        """
        Extracting inputfile and return all tokens in a list. Each BOS we mark with as many <s> as n_gram_size,
        transmitted by the user. This is necessary for the generation of ngrams. Each EOS we mark with </s>.
        :param path_to_document:    Path to input file, str
        :param n_gram_size:         size of n_grams on base of which we calculate the probabilites for follower tokens,
                                    int
        :return:                    Tokenlist, lst
        """
        token_lst = ['<s>' for i in range(0,n_gram_size-1)]
        with open(path_to_document, 'r') as f:
            for line in f:
                for i in range(0,n_gram_size-1):
                    token_lst.append('<s>')
                for word in line.split():
                    token_lst.append(word)
                token_lst.append('</s>')
        return(token_lst)

    @staticmethod
    def get_ngram_freqs(tokens: List[str], n_gram_size: int):
        """
        :param tokens:              All tokens from the inputfile in a list, lst
        :param n_gram_size:         size of n_grams on base of which we calculate the probabilites for follower tokens,
                                    int
        :return:                    Returns the ngrams the inputlines contain, alongside their frequency, as a two-
                                    dimensional dictionary, Dict[tuple, Dict[str, float]]
        """
        n_gram_lst = list(ngrams(tokens, n_gram_size))
        freq = defaultdict(lambda: defaultdict(lambda: 0.0))
        i = 0
        for ngram in n_gram_lst:
            hist_tup = tuple(n_gram_lst[i][0:n_gram_size-1])
            freq[hist_tup][n_gram_lst[i][n_gram_size-1]] += 1
            i += 1
        return freq

    def generate_sentence(self, freq: Dict[tuple, Dict[str, float]], n_gram_size: int):
        """
        :param freq:                The ngrams the inputlines contain, alongside their frequency, as a two-dimensional
                                    dictionary, Dict[tuple, Dict[str, float]]
        :param n_gram_size:         size of n_grams on base of which we calculate the probabilites for follower tokens,
                                    int
        :return:                    A generated songtext line, lst
        """
        sum_of_fitness = 0
        hist_lst = ['<s>' for i in range(0,n_gram_size-1)]
        hist_tup = tuple(hist_lst)
        gen_seq = []
        follower = ['<s>']
        "calculate 'fitness' of each follower-item"
        while (follower != '</s>'):
            for key, fitness in freq[hist_tup].items():
                sum_of_fitness += fitness
            "If we have n_gram_size = 1, we can't make use of any frequencies, we just chose tokens randomly out of" \
            "the input files."
            if n_gram_size == 1:
                follower = random.choices([freq for freq in freq[hist_tup].keys()], k=1)[0]
            else:
                follower = random.choices([freq for freq in freq[hist_tup].keys()],
                                          [fitness/sum_of_fitness for fitness in freq[hist_tup].values()], k=1)[0]
            hist_lst = [hist_lst[i] for i in range(1,n_gram_size-1)]
            hist_lst.append(follower)
            hist_tup = tuple(hist_lst)
            if follower != '</s>':
                gen_seq.append(follower)
        return gen_seq
