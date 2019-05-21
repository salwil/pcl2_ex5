#!/usr/bin/env python3
#-*- coding: utf-8 -*-
# Roger Rüttimann rroger 02-914-471

from collections import defaultdict
from typing import Dict, Iterable, List
import math
import operator
from sklearn.metrics import recall_score, accuracy_score, f1_score, precision_score
import itertools
from collections import Counter
from string import punctuation
import nltk

class LyricsClassifier:
    """Implements a Naïve Bayes model for song lyrics classification."""

    def __init__(self, train_data: Dict[str, Iterable[str]]):
        """Initialize and train the model.

        Args:
            train_data: A dict mapping labels to iterables of lines of
                        (e.g. a file object).
        """
        label_counts = {}
        label_feature_value_counts = {}
        # store the different seen values for each feature (across the entire
        # training set, independent of label), needed for smoothing
        feature_values = defaultdict(lambda: set())

        for label, lines in train_data.items():
            # calculate raw counts given this label
            label_count = 0
            feature_value_counts = defaultdict(lambda: defaultdict(int))
            for line in lines:
                label_count += 1
                features = self._extract_features(line)
                for feature_id, value in features.items():
                    feature_value_counts[feature_id][value] += 1
                    feature_values[feature_id].add(value)
            label_counts[label] = label_count
            label_feature_value_counts[label] = feature_value_counts

        self.label_counts = label_counts
        total_label_lines = sum(label_counts.values())
        self.label_a_priori = {}
        self.a_posteriori = {}
        length_feature_values = len(feature_values)
        for label in label_counts:
            self.label_a_priori[label] = math.log(label_counts[label] / total_label_lines)
            self.a_posteriori[label] = {}
            for feature in feature_values.keys():
                self.a_posteriori[label][feature] = defaultdict(float)
                total_feature = sum(label_feature_value_counts[label][feature].values())
                for value, count in label_feature_value_counts[label][feature].items():
                    self.a_posteriori[label][feature][value] = math.log((count + 1) / (total_feature + length_feature_values))

    @staticmethod
    def split_data_for_cross_validation(train_data: Dict[str, str], parts: int) -> Dict[str, List[Dict[str, str]]]:
        """Splits data and returns a dict containing label and file names of cross combined test and evaluation data

        Args:
            train_data: A dict mapping labels to file paths

            parts: Number of parts. One part is evaluation data, the rest is training data

        Returns:
            A dict mapping label to the file names of evaluation and training data
        """
        result = {}
        for label, file_name in train_data.items():
            combinations = []
            with open(file_name, 'r', encoding='UTF8') as data:
                lines = data.readlines()
                length_data = len(lines)
                length_part = math.ceil(length_data/parts)
                for index in range(parts):
                    for test_index in range(parts):
                        test_filename = f"./test{label}_{test_index}.eval"
                        with open(test_filename, 'w', encoding='UTF8') as test_file:
                            test_file.writelines(lines[test_index * length_part : (test_index + 1) * length_part])

                        training_filename = f"./test{label}_{test_index}.train"
                        with open(training_filename, 'w', encoding='UTF8') as training_file:
                            if test_index == 0:
                                training_file.writelines(lines[length_part:])
                            elif test_index == parts - 1:
                                training_file.writelines(lines[0: test_index *length_part])
                            else:
                                training_file.writelines(lines[0 : test_index * length_part])
                                training_file.writelines(lines[(test_index + 1) * length_part : ])
                        combinations.append({ 'eval': test_filename, 'train': training_filename})
            result[label] = combinations
        return result

    @staticmethod
    def _extract_features(line: str) -> Dict:
        """Return a dict containing features values extracted from a line.

        Args:
            line: A line of song lyrics.

        Returns:
            A dict mapping feature IDs to feature values.
        """
        return {
            'pray':    'pray' in line,
            'ing':    'ing' in line,
            're':    "'re" in line.split(),
            'feel': 'feel' in line,
            'exclamation': '!' in line,
            'question': '?' in line,
            'na': 'na' in line,
            'know': 'know' in line,
            'is': 'is' in line,
            'it': 'it' in line,
            'bracket1': '(' in line,
            'bracket2': ')' in line,
            'love': 'love' in line,
            'we': 'we' in line.split(),
            'me': 'me' in line.split(),
            'm': "'m" in line.split(),
            'one': 'one' in line.split(),
            'come': 'come' in line.split(),
            'sign': "'" in line.split(),
            'grunts':  any(grunt in line for grunt in ['oh', 'ah', 'uh']),
            'tokens':  len(line.split()),
            'charset': len(set(line)) // 4,
            'number_of_consonants': sum(line.count(c) for c in "bcdfghjklmnpqrstvwxyz"),
            'number_of_vowels': sum(line.count(c) for c in "aeiou"),


        }

    def _probability(self, label: str, features: Dict) -> float:
        """Return P(label|features) using the Naïve Bayes assumption.

        Args:
            label: The label.
            features: A dict mapping feature IDs to feature values.

        Returns:
            The non-logarithmic probability of `label` given `features`.
        """
        sum_features = 0
        for feature, value in features.items():
            sum_features += self.a_posteriori[label][feature][value]
        return math.exp(self.label_a_priori[label] + sum_features)

    def predict_label(self, line: str) -> str:
        """Return the most probable prediction by the model.

        Args:
            line: The line to classify.

        Returns:
            The most probable label according to Naïve Bayes.
        """
        predictions = {}
        for label in self.label_a_priori.keys():
            predictions[label] = self._probability(label, self._extract_features(line))
        return max(predictions.items(), key=operator.itemgetter(1))[0]


    def evaluate(self, test_data: Dict[str, Iterable[str]]):
        """Evaluate the model and print recall, precision, accuracy and f1-score.

        Args:
            test_data: A dict mapping labels to iterables of lines
                       (e.g. a file object).
        """
        label_true = []
        label_pred = []
        result = {}
        for label, lines in test_data.items():
            for line in lines:
                label_true.append(label)
                label_pred.append(self.predict_label(line))

        result['recall'] = recall_score(label_true, label_pred, average='micro')
        result['accuracy'] = accuracy_score(label_true, label_pred)
        result['precision'] = precision_score(label_true, label_pred, average='micro')
        result['f1'] = f1_score(label_true, label_pred, average='micro')

        print(result)
        return result


if __name__ == '__main__':
    lc = None
    data_parts = 5
    file_names = LyricsClassifier.split_data_for_cross_validation({'bobo': './djbobo.train', 'jackson': './michaeljackson.train'}, 5)
    stats = []
    for index in range(data_parts):
        with open(file_names['bobo'][index]['train'], 'r', encoding='UTF8') as bobo_training_data, open(file_names['jackson'][index]['train'], 'r', encoding='UTF8') as jackson_training_data:
            lc = LyricsClassifier( train_data= { 'bobo': bobo_training_data, 'jackson': jackson_training_data } )
        with open(file_names['bobo'][index]['eval'], 'r', encoding='UTF8') as bobo_eval_data, open(file_names['jackson'][index]['eval'], 'r', encoding='UTF8') as jackson_eval_data:
            stats.append(lc.evaluate( test_data= { 'bobo': bobo_eval_data, 'jackson': jackson_eval_data } ))

    avgDict = defaultdict(float)
    for stat in stats:
        for key, value in stat.items():
            avgDict[key] += value
    for key, value in avgDict.items():
        avgDict[key] = value / data_parts
    print(avgDict)

    def content_text(text):
        stopwords = set(nltk.corpus.stopwords.words('english'))
        with_stp = Counter()
        without_stp  = Counter()
        for line in text:
            for word in line.split():
                # update count off all words in the line that are in stopwords
                word = word.lower()
                if word in stopwords:
                    with_stp.update([word])
                else:
                # update count off all words in the line that are not in stopwords
                    without_stp.update([word])
        # return a list with top ten most common words from each
        return [k for k,_ in with_stp.most_common(15)],[y for y,_ in without_stp.most_common(15)]

    with open('./djbobo.train', 'r', encoding='utf8') as bobo:
        print(content_text(bobo))

    with open('./michaeljackson.train', 'r', encoding='utf8') as jackson:
            print(content_text(jackson))
