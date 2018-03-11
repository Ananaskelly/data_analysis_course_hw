import numpy as np
import scipy.stats as stats


def norm_dist(y):
    return 1/np.sqrt(2*np.pi)*np.exp(-y**2/2)


def ks_test_uniform(v, alpha):
    """
    Compare given sample with a uniform distribution.

    :param v: given sample
    :param alpha: significance level
    :return:
    """
    v = np.asarray(v).flatten()
    n = v.shape[0]

    # эмпирическая функция распределения
    v_var = np.sort(v)

    # теоритеческая
    v_emp = np.arange(0, n)/n
    diff = np.abs(v_var - v_emp)
    d = np.max(diff)

    lmbd = d * np.sqrt(n)
    val = stats.ksone.ppf(1 - alpha, n)

    print('lambda estimated: {}, table: {}'.format(lmbd, val))
    if lmbd < val:
        print('не можем опровергнуть')
    else:
        print('отвергаем')


def chi_sq_test(v, alpha, k):
    """
    Compare given sample with a uniform distribution.

    :param v: given sample
    :param alpha: significance level
    :param k:
    :return:
    """
    v = np.asarray(v).flatten()
    n = v.shape[0]

    hist, bins = np.histogram(v, k)
    dlt = bins[1] - bins[0]
    v_mean = np.sum(hist*bins[1:])/n
    v_mean_sq = np.sum(hist*bins[1:]**2)/n
    v_std = np.sqrt(v_mean_sq - v_mean**2)
    print('mean: {}, std: {}'.format(v_mean, v_std))

    v_t = list(map(lambda x: n*dlt/v_std*norm_dist((x - v_mean)/v_std), bins[1:]))
    v_t = np.array(v_t)
    chi_sq = np.sum(np.divide((hist - v_t)**2, v_t))

chi_sq_test(np.random.randn(42), 0.005, 7)
