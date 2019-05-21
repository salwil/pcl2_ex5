#!/usr/bin/env python3
#-*- coding: utf-8 -*-
# Roger Rüttimann rroger 02-914-471'

import click
from typing import Dict, Iterable, List
from classify import LyricsClassifier
from generate import NGramModel

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.group()
def lyrics():
  pass

@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('train_file', nargs=-1, type=click.Path())
@click.option("--text", default=None, help="text to use as input (otherwise, standard input is used)")
@click.option("--evaluate", is_flag=True, help="evaluate performance on the training set")
def classify(train_file, text, evaluate):
  """
    Classify lyrics using Naïve Bayes.

    positional arguments:

      train_file  files containing training data for each label; filename format: <label>.train
  """

  train_data = _open_train_data(train_file)
  lyrics_classifier = LyricsClassifier(train_data=train_data)
  _close_files(train_data)

  if evaluate:
    train_data = _open_train_data(train_file)
    lyrics_classifier.evaluate(test_data=train_data)
    _close_files(train_data)
  else:
    if text == None:
      while(True):
        text = click.prompt('', type=str, prompt_suffix='')
        click.echo(lyrics_classifier.predict_label(text))
    else:
      click.echo(lyrics_classifier.predict_label(text))

def _close_files(train_data: Dict[str, Iterable[str]]):
  """Close open files in train_data

    arguments:

      train_data: A dict mapping labels to iterables of lines of
                              (e.g. a file object).
  """

  for label in train_data:
    train_data[label].close()

def _open_train_data(train_file: Iterable[str]) -> Dict[str, Iterable[str]]:
  """Open files in train_file and return Dict of label with open files as values.

    arguments:

      train_file  files containing training data for each label; filename format: <label>.train
  """

  train_data = {}
  for entry in train_file:
    label = entry.replace('.train', '')
    train_data[label] = open(entry, 'r', encoding='UTF8')
  return train_data


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('filename')
@click.option("-l", default=1, help="The number of lines to generate.")
@click.option("-n", prompt="Please enter the ngram-size from 1 to 5", type=click.IntRange(1,5), help="Ngram size. ")
@click.option("--forever", is_flag=True, default=False, help="If we should never stop creating songlines anymore.")
def generate(filename, l, n, forever):
    ngram_model = NGramModel(filename, l, n, forever)

lyrics.add_command(classify)
lyrics.add_command(generate)


if __name__ == '__main__':
    lyrics()
