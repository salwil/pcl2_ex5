#!/usr/bin/env python3

import os
import glob
import random
from math import log10
from collections import defaultdict
from typing import Dict, List
from nltk.util import ngrams
from numpy.random import choice as np
import random

"""Fragen:
    - defaultdic gibt für jeden Schlüssel, der nicht abgelegt ist den Standardwert, der dafür implementiert wurde (z.B.
      mithilfe einer Lambdafunktion) zurück. Eignet sich gut für Wahrscheinlichkeiten.
    - Ich fahre nach Hause. Ich gehe nach Hause. --> Wahrscheinlichkeiten für auf "Ich" folgende Wörter berechnen
    - Satzanfang nehmen wir nicht mit, Satzende schon (weil wir Satzende zur Generierung benötigen)
    - ngrams = defaultdic(lambda: defaultdict(lambda:1/Vokabulargroesse))
      --> ngrams[('<s>', '<s>')]['Ich'] * ngrams [('<s>', 'Ich')]['gehe'] * ngrams[('Ich', 'gehe')]['nach'] 
      = P für "Ich gehe nach".
    - ngrams = defaultdict(lambda: defaultdict(p: defaultdict(float) for p in range (0, self.n)))
    - wir müssen die möglichen folgenden Elemente zufällig aber multipliziert mit ihrer Wahrscheinlichkeit ziehen.
    - n --> History enthält n-1 Elemente, aber wir gehen tokenweise voran bei der Zeilengenerierung.
"""

class NGramModel:
    def __init__(self, path_to_document: str, n: int):
        self.input = self.extract_input_file(path_to_document, n)
        self._ngram_freqs = self.get_ngram_freqs(self.input, n)
        print(self.generate_sentence(self._ngram_freqs, n))

    def extract_input_file(self, path_to_document: str, n: int):
        token_lst = ['<s>' for i in range(0,n-1)]
        with open(path_to_document, 'r') as f:
            for line in f:
                for i in range(0,n-1):
                    token_lst.append('<s>')
                for word in line.split():
                    token_lst.append(word)
                token_lst.append('</s>')
        return(token_lst)

    @staticmethod
    def get_ngram_freqs(tokens: List[str], n: int):
        """
        Returns the ngrams the inputlines contain, alongside their frequency, as a dictionary.
        """
        n_gram_lst = list(ngrams(tokens, n))
        freq = defaultdict(lambda: defaultdict(lambda: 0.0))
        i = 0

        for ngram in n_gram_lst:
            hist_tup = tuple(n_gram_lst[i][0:n-1])
            freq[hist_tup][n_gram_lst[i][n-1]] += 1
            i += 1
        return freq

    def generate_sentence(self, freq: Dict[tuple, Dict[str, float]], n: int):
        sum_of_fitness = 0
        hist_lst = ['<s>' for i in range(0,n-1)]
        hist_tup = tuple(hist_lst)
        gen_seq = []
        follower = ['<s>']
        """
        calculate 'fitness' of each follower-item
        """
        while (follower != '</s>'):
            for key, fitness in freq[hist_tup].items():
                sum_of_fitness += fitness
            follower = random.choices([freq for freq in freq[hist_tup].keys()],
                                     [fitness/sum_of_fitness for fitness in freq[hist_tup].values()], k=1)[0]
            hist_lst = [hist_lst[i] for i in range(1,n-1)]
            hist_lst.append(follower)
            hist_tup = tuple(hist_lst)
            gen_seq.append(follower)
        return gen_seq


def main():
    ngram_model = NGramModel('michaeljackson.train', 5)


if __name__ == '__main__':
    main()