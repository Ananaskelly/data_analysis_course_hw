import pandas as pd


def fix_csv(path):
    data = pd.read_csv(path,
                       sep='\\s+|\\t+|:|\'|,"|,|"',
                       header=None,
                       engine='python')
    data.to_csv('output.csv', sep=',', index=False)
