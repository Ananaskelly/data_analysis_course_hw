import nltk
from nltk import Nonterminal, nonterminals, Production, CFG


def run():
    cool_grammar = nltk.CFG.fromstring("""
    S -> NP VP
    PP -> Det N
    NP -> Det N | PP PP | 'я'
    VP -> V NP | VP PP
    Det -> 'его' | 'своими'
    N -> 'семью' | 'глазами'
    V -> 'видел'
    """)

    sent = ['я', 'видел', 'его', 'семью', 'своими', 'глазами']
    parser = nltk.ChartParser(cool_grammar)

    for tree in parser.parse(sent):
        print(tree)
