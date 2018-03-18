import nltk
import pymorphy2 as pm
import re
from nltk.collocations import *


def read_file(file_name='../../data/balabanov.txt'):
    file = open(file_name, 'r', encoding='utf-8')
    return file.read()


def nltk_tokenize(raw):
    tokens = nltk.word_tokenize(raw)
    text = nltk.Text(tokens)
    return tokens, text


def custom_tokenize(raw):
    raw_low = raw.lower()

    symbols_to_remove = "\"«»“”!#$%&\'()*+,/:;<=>?@[\\]^_`{|}~—"

    clear_txt = raw_low.translate(str.maketrans("", "", symbols_to_remove))
    clear_txt = re.sub(r'([^. ]{2,})\.', '\g<1>', clear_txt)
    return re.split(r' |\n', clear_txt)


def part_a(nltkText, n=50):
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    trigram_measures = nltk.collocations.TrigramAssocMeasures()

    finder_bi = BigramCollocationFinder.from_words(nltkText)
    finder_thr = TrigramCollocationFinder.from_words(nltkText)

    print(finder_bi.nbest(bigram_measures.pmi, n))
    print(finder_thr.nbest(trigram_measures.pmi, n))


def part_b(tokens):
    morph = pm.MorphAnalyzer()
    tokens_nf = []
    for token in tokens:
        tokens_nf.append(morph.parse(token)[0].normal_form)
    return nltk.Text(tokens_nf)


def part_v(raw):
    tokens = custom_tokenize(raw)
    return tokens
