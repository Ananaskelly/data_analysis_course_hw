import numpy as np
import scipy.stats as stats


def calc():
    n = 36
    alpha = 0.005
    mean_1 = 4.51
    mean_2 = 6.28
    std_1 = 1.98
    std_2 = 2.54

    t = (mean_1 - mean_2)/np.sqrt(std_1**2/n + std_2**2/n)
    # хотим альтернативу mean_1 < mean_2
    q_val = stats.t.ppf(q=alpha, df=n+n-1)
    print('t_value: {}'.format(t))
    print('q_value: {}'.format(q_val))

    if t < q_val:
        print('Отвергаем H0')
    else:
        print('Не можем опровергнуть H0')
