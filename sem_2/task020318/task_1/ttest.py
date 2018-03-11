import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import math


def calc(alpha):
    np.random.seed(6)

    # генерируются две выборки, распределенные по Пуассону

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html
    population_ages1 = stats.poisson.rvs(loc=18, mu=35, size=150000)
    population_ages2 = stats.poisson.rvs(loc=18, mu=10, size=100000)
    population_ages = np.concatenate((population_ages1, population_ages2))

    minnesota_ages1 = stats.poisson.rvs(loc=18, mu=30, size=30)
    minnesota_ages2 = stats.poisson.rvs(loc=18, mu=10, size=20)
    minnesota_ages = np.concatenate((minnesota_ages1, minnesota_ages2))

    # round(x, n)
    print(round(population_ages.mean(), 2)) # генеральное среднее
    print(round(minnesota_ages.mean(), 2)) # выборочное среднее

    # одновыборочный t-критерий Стьюдента, используется, когда известны параметры ГС,
    t_value, p_value = stats.ttest_1samp(a=minnesota_ages, popmean=population_ages.mean())
    # рассчитываем t-критерий
    print(t_value, p_value)

    # проверим, какая часть значений лежит вне нашего доверительного интервала
    # при заданных уровне значимости alpha и числе степеней свободы 49 (minnesota_ages - 1)

    t_value_left = stats.t.ppf(q=alpha*0.5, df=49)
    print("t-value (left) = %g" %round(t_value_left, 5))

    t_value_right = stats.t.ppf(q=1-alpha*0.5,  #
                df=49)  # число степеней свободы
    print("t-value (right) = %g" %round(t_value_right, 5))

    p_val = stats.t.cdf(x= -2.5742,      # значение t-критерия, рассчитанное по эмпир.данным
                   df= 49) * 2   # т.к. двусторонний критерий
    print("p-value = %g" %round(p_val, 3))


#######################################################################################################################


def print_answer():
    print('Answer: не можем отклонить гипотезу H0 на заданном уровне значимости alpha=0.001')
