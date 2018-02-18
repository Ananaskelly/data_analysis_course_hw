import pandas as pd


def estimate(path_to_file):
    df = pd.read_csv(path_to_file, sep=',', header=None)
    pd.set_option('display.width', 100)
    pd.set_option('precision', 3)
    description = df.describe()
    print('Mode for column {}:\n {}'.format(1, df[[1]].mode()))
    print('Description:')
    print(description)
