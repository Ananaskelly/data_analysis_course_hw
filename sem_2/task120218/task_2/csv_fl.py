import pandas as pd


def fix_csv(path):
    data = pd.read_csv(path,
                       sep=',',
                       header=0,
                       engine='python')
    data[['FuelEconomy']] = data[['FuelEconomy']].apply(pd.to_numeric, errors='coerce')
    data = data.fillna(data.mean())
    data.to_csv('output.csv', index=False)
