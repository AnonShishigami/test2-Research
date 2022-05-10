import numpy as np
from multiprocessing import Process, Queue
from ammlib import Logistic, LogisticTools, Market,\
                   BaseOracle, PerfectOracle, LaggedOracle,\
                   LiquidityProviderCstDelta, LiquidityProviderAMMSqrt, LiquidityProviderBestClosedForm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def MonteCarlo(currencies_params, sizes, log_params, lp_params, simul_params, nb_MCs, seed, q):

    currencies, init_swap_price_01, mu, sigma = currencies_params
    nb_sizes = sizes.shape[0]
    lambda_, a, b = log_params
    intensity_functions_01_object = [Logistic(lambda_, a, b) for _ in range(nb_sizes)]
    intensity_functions_10_object = [Logistic(lambda_, a, b) for _ in range(nb_sizes)]
    market = Market(currencies, init_swap_price_01, mu, sigma, sizes, intensity_functions_01_object,
                    intensity_functions_10_object)
    typo, initial_inventories, initial_cash, extra_params = lp_params
    dt_sim, T_sim = simul_params

    if typo == 'POcst':
        delta = extra_params
        lp = LiquidityProviderCstDelta('POcst%d' % delta, initial_inventories.copy(), initial_cash, market, PerfectOracle(), delta * 1e-4)
    elif typo == 'LOcst':
        delta = extra_params
        lp = LiquidityProviderCstDelta('LOcst%d' % delta, initial_inventories.copy(), initial_cash, market,
                                       LaggedOracle(10./(24.*60.*60.)), delta * 1e-4)
    elif typo == 'ammsqrt':
        delta = extra_params
        lp = LiquidityProviderAMMSqrt('cfmm%d' % delta, initial_inventories.copy(), initial_cash, market, BaseOracle(), delta * 1e-4)
    elif typo == 'POmyopic':
        delta = extra_params
        lp = LiquidityProviderCstDelta('myopic', initial_inventories.copy(), initial_cash, market,
                                       PerfectOracle(), delta * 1e-4)
    elif typo == 'PObcf':
        gamma = extra_params
        lp = LiquidityProviderBestClosedForm('PObcf%.0e' % gamma, initial_inventories.copy(), initial_cash, market,
                                             PerfectOracle(), gamma)
    elif typo == 'LObcf':
        gamma = extra_params
        lp = LiquidityProviderBestClosedForm('LObcf%.0e' % gamma, initial_inventories.copy(), initial_cash, market,
                                             LaggedOracle(10./(24.*60.*60.)), gamma)


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
    init_swap_price_01 = 3000. / 40000.
    mu = 0.
    scale = 1. / 252.
    sigma = 0.2 * np.sqrt(scale)

    currencies_params = (currencies, init_swap_price_01, mu, sigma)

    sizes = np.array([1. / 40.])
    nb_sizes = sizes.shape[0]

    lambda_ = 45.
    a = -1.8
    b = 1300.

    log_params = lambda_, a, b

    initial_inventories = 20. * np.array([1., 10. * 4. / 3.])
    initial_cash = 0.

    dt_sim = 10. / (24. * 60. * 60.)
    T_sim = 1.
    simul_params = (dt_sim, T_sim)

    nb_MCs = 100
    seed = 42

    q = Queue()

    names = []
    means = []
    stdevs = []
    colors = []

    jobs = []
    typo = 'POcst'
    deltas = np.arange(1, 30, 2)

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
    typo = 'LOcst'
    deltas = np.arange(1, 30, 2)

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
        colors.append('cyan')

    jobs = []
    typo = 'POmyopic'
    lt = LogisticTools(lambda_, a, b)
    mq = lt.delta(0.) * 1e4
    print(mq)

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
    deltas = np.arange(1, 30, 2)
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

    gammas = [0., 1., 5., 10., 50., 100., 500., 1000., 5000., 10000.]

    jobs = []
    typo = 'PObcf'

    for gamma in gammas:
        lp_params = (typo, initial_inventories, initial_cash, gamma)
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

    jobs = []
    typo = 'LObcf'

    for gamma in gammas:
        lp_params = (typo, initial_inventories, initial_cash, gamma)
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
        colors.append('orange')

    plt.rcParams["figure.figsize"] = [16, 9]
    fig, ax = plt.subplots(1, 1)

    ax.scatter(np.array(stdevs), np.array(means), c=colors, alpha=0.5)
    ax.set_xlabel('Standard deviation of PnL - PnL Hodl (in %s) after %.1f day(s)' % (currencies[0], T_sim))
    ax.set_ylabel('Mean of PnL - PnL Hodl (in %s) after %.1f day(s)' % (currencies[0], T_sim))
    ax.set_title('Statistics of PnL - PnL Hodl (in %s) after %.1f day(s) for different LP strategies' % (currencies[0], T_sim))

    for name, mean, stdev in zip(names, means, stdevs):
        ax.annotate(name, (stdev, mean))

    plt.savefig('frontier_basics.pdf')
    plt.show()
