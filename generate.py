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
    - müssen wir die Werte gehasht verarbeiten? Falls ja, wie kann man Daten wieder aus Hash extrahieren?
    - Können wir die Codebeispiele jeweils raufladen im OLAT?
    - Inputformat???
    - Können wir davon ausgehen, dass Satzzeichen separate Tokens sind? Oder haben wir ein Inputformat mit Satzanfangs- 
      und Satzende-Tags?
    - zweidim. Dictionnaire: wie geht das, wenn Dict unhashable ist?
    - defaultdic gibt für jeden Schlüssel, der nicht abgelegt ist den Standardwert, der dafür implementiert wurde (z.B.
      mithilfe einer Lambdafunktion) zurück. Eignet sich gut für Wahrscheinlichkeiten.
    - Ich fahre nach Hause. Ich gehe nach Hause. --> Wahrscheinlichkeiten für auf "Ich" folgende Wörter berechnen, inkl.
      Smoothing. 
    - Smoothing: Vokabulargrösse mitbeachten. P(fahre|Ich) = (1+1)/2+8 = 1/5, 8 ist Vokabulargrösse, die auch noch mit-
      beachtet werden. Schwäche: 60% der Wahrscheinlichkeitsmasse der Wörter, die auf "Ich" folgen können, sind undefi-
      niert.
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
        token_lst = []
        with open(path_to_document, 'r') as f:
            for line in f:
                token_lst = ['<s>' for i in range(0,n-1)]
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
        print(n_gram_lst)
        freq = defaultdict(lambda: defaultdict(lambda: 0.0))
        i = 0

        for ngram in n_gram_lst:
            hist_tup = tuple(n_gram_lst[i][0:n-1])
            freq[hist_tup][n_gram_lst[i][n-1]] += 1
            i += 1
        print(freq)

        return freq

    def generate_sentence(self, freq: Dict[tuple, Dict[str, float]], n: int):
        sum_of_fitness = 0
        hist_lst = ['<s>' for i in range(0,n-1)]
        hist_tup = tuple(hist_lst)
        print(hist_tup)
        gen_seq = []
        follower = ['<s>']
        """
        calculate 'fitness' of each follower-item
        """
        while (follower != '</s>'):
            print(hist_tup)
            for key, fitness in freq[hist_tup].items():
                print(freq[hist_tup])
                sum_of_fitness += fitness
                print(key)
                print(fitness)
            follower = random.choices([freq for freq in freq[hist_tup].keys()],
                                     [fitness/sum_of_fitness for fitness in freq[hist_tup].values()], k=1)[0]
            hist_lst = [hist_lst[i] for i in range(1,n-1)]
            print('Hist_list: ', hist_lst)
            hist_lst.append(follower)
            hist_tup = tuple(hist_lst)
            print('Hist_tup: ', hist_tup)
            gen_seq.append(follower)
            print(key, sum_of_fitness)
        print(gen_seq)


def main():
    ngram_model = NGramModel('michaeljackson.train', 2)


if __name__ == '__main__':
    main()