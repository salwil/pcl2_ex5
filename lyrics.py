#!/usr/bin/env python3
#-*- coding: utf-8 -*-
# Roger Rüttimann rroger 02-914-471'

import click
#from classify import LyricsClassifier
from generate import NGramModel

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.group()
def lyrics():
  pass

@click.command(context_settings=CONTEXT_SETTINGS)
# @click.option('-h', '--help', 'string')
@click.argument('train_file')
def classify(train_file):
  """
    Classify lyrics using Naïve Bayes.

    positional arguments:

      train_file  files containing training data for each label; filename format: <label>.train
  """
  click.echo('classify')


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('filename')
@click.option("-l", default=1, help="The number of lines to generate.")
@click.option("-n", prompt="Please enter the ngram-size from 1 to 5", type=click.IntRange(1,5), help="Ngram size. ")
@click.option("--forever", is_flag=True, default=False, help="If we should never stop creating songlines anymore.")
def generate(filename, l, n, forever):
    print(l)
    print(n)
    ngram_model = NGramModel(filename, l, n, forever)

lyrics.add_command(classify)
lyrics.add_command(generate)


if __name__ == '__main__':
    lyrics()
