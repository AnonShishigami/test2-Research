import numpy as np
from multiprocessing import Process, Queue
from ammlib import Logistic, Market, LiquidityProviderCstDelta, LiquidityProviderAMMSqrt, LogisticTools
from ammlib.lpopt import LiquidityProviderHJB, NumericalParams
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def MonteCarlo(currencies_params, sizes, log_params, lp_params, simul_params, nb_MCs, seed, q):

    currencies, init_prices, mu, Sigma = currencies_params
    nb_sizes = sizes.shape[0]
    lambda_, a, b = log_params
    intensity_functions_01_object = [Logistic(lambda_, a, b) for _ in range(nb_sizes)]
    intensity_functions_10_object = [Logistic(lambda_, a, b) for _ in range(nb_sizes)]
    market = Market(currencies, init_prices, mu, Sigma, sizes, intensity_functions_01_object,
                    intensity_functions_10_object)
    typo, initial_inventories, initial_cash, extra_params = lp_params
    dt_sim, T_sim = simul_params

    if typo == 'cst':
        delta = extra_params
        lp = LiquidityProviderCstDelta('cst%d' % delta, initial_inventories.copy(), initial_cash, market, delta * 1e-4)
    elif typo == 'ammsqrt':
        delta = extra_params
        lp = LiquidityProviderAMMSqrt('cfmm%d' % delta, initial_inventories.copy(), initial_cash, market,delta * 1e-4)
    elif typo == 'myopic':
        delta = extra_params
        lp = LiquidityProviderCstDelta('myopic', initial_inventories.copy(), initial_cash, market, delta * 1e-4)
    elif typo == 'hjb':
        T, nb_t, V_min, V_max, gamma = extra_params
        num_params = NumericalParams(T, nb_t, V_min, V_max)
        lp = LiquidityProviderHJB('HJB%.0e' % gamma, initial_inventories.copy(),
                                  initial_cash, market, gamma, num_params)
    else:
        return

    print('Starting ' + lp.name)

    pnls = np.zeros(nb_MCs)
    np.random.seed(seed)

    for j in range(nb_MCs):
        if j % 10 == 0:
            print(lp.name + ': ' + str(j))
        lp.reset()
        res = market.simulate(dt_sim, T_sim, lp)
        pnls[j] = res.pnl[-1]

    q.put((lp.name, np.mean(pnls), np.std(pnls)))
    print('Done with ' + lp.name)


if __name__ == '__main__':

    currencies = ['BTC', 'ETH']
    init_prices = np.array([40000., 3000.])
    mu = np.zeros(2)
    scale = 1./252.
    volatilities = np.array([0.4, 0.4])
    rho = 0.9
    Sigma = scale * np.array([[volatilities[0]**2, rho * volatilities[0] * volatilities[1]],
                  [rho * volatilities[0] * volatilities[1], volatilities[1]**2]])

    currencies_params = (currencies, init_prices, mu, Sigma)

    sizes = np.array([1000., 2000.])

    lambda_ = 100.
    a = 0.
    b = 2e3
    log_params = lambda_, a, b

    initial_inventories = 20. * np.array([1., 10. * 4. / 3.])
    initial_cash = 0.

    dt_sim = 10. / (24. * 60. * 60.)
    T_sim = 1.
    simul_params = (dt_sim, T_sim)

    nb_MCs = 50
    seed = 42

    q = Queue()

    names = []
    means = []
    stdevs = []
    colors = []

    jobs = []
    typo = 'cst'
    deltas = [] #np.arange(1, 25, 2)

    for delta in deltas:
        lp_params = (typo, initial_inventories, initial_cash, delta)
        job = Process(target=MonteCarlo, args=(currencies_params, sizes, log_params, lp_params, simul_params, nb_MCs, seed, q))
        job.start()
        jobs.append(job)

    for job in jobs:
        job.join()

    for delta in deltas:
        name, mean, stdev = q.get()
        names.append(name)
        means.append(mean)
        stdevs.append(stdev)
        colors.append('blue')

    jobs = []
    typo = 'myopic'
    lt = LogisticTools(lambda_, a, b)
    mq = lt.myopic_quote * 1e4

    lp_params = (typo, initial_inventories, initial_cash, mq)
    job = Process(target=MonteCarlo,
                  args=(currencies_params, sizes, log_params, lp_params, simul_params, nb_MCs, seed, q))
    job.start()
    jobs.append(job)

    for job in jobs:
        job.join()

    name, mean, stdev = q.get()
    names.append(name)
    means.append(mean)
    stdevs.append(stdev)
    colors.append('black')

    jobs = []
    typo = 'ammsqrt'
    deltas = [] #np.arange(1, 25, 1)
    for delta in deltas:
        lp_params = (typo, initial_inventories, initial_cash, delta)
        job = Process(target=MonteCarlo,
                      args=(currencies_params, sizes, log_params, lp_params, simul_params, nb_MCs, seed, q))
        job.start()
        jobs.append(job)

    for job in jobs:
        job.join()

    for delta in deltas:
        name, mean, stdev = q.get()
        names.append(name)
        means.append(mean)
        stdevs.append(stdev)
        colors.append('red')

    jobs = []
    typo = 'hjb'
    T = 0.5
    nb_t = 101
    Vs_min = initial_inventories * init_prices - 30. * sizes[0]
    Vs_max = initial_inventories * init_prices + 30. * sizes[0]
    gammas = [0., 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
    for gamma in gammas:
        extra_params = (T, nb_t, Vs_min, Vs_max, gamma)
        lp_params = (typo, initial_inventories, initial_cash, extra_params)
        job = Process(target=MonteCarlo,
                      args=(currencies_params, sizes, log_params, lp_params, simul_params, nb_MCs, seed, q))
        job.start()
        jobs.append(job)

    for job in jobs:
        job.join()

    for gamma in gammas:
        name, mean, stdev = q.get()
        names.append(name)
        means.append(mean)
        stdevs.append(stdev)
        colors.append('green')

    plt.rcParams["figure.figsize"] = [16, 9]
    fig, ax = plt.subplots(1, 1)

    ax.scatter(np.array(stdevs), np.array(means), c=colors)
    ax.set_xlabel('Standard deviation of PnL - PnL Hodl after %.1f day(s)' % T_sim)
    ax.set_ylabel('Mean of PnL - PnL Hodl after %.1f day(s)' % T_sim)
    ax.set_title('Statistics of PnL - PnL Hodl after %.1f day(s) for different LP strategies' % T_sim)

    for name, mean, stdev in zip(names, means, stdevs):
        ax.annotate(name, (stdev, mean))

    plt.show()
