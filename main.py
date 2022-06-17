from multiprocessing import Process, Queue
import warnings

import matplotlib.pyplot as plt
import numpy as np

from ammlib import Logistic, LogisticTools, Market,\
    BaseOracle, PerfectOracle, LaggedOracle, SparseOracle,\
    LiquidityProviderCstDelta, LiquidityProviderAMMSqrt, LiquidityProviderBestClosedForm,\
    LiquidityProviderSwaapV1


warnings.filterwarnings("ignore")

NUM_SEC_PER_DAY = 24 * 60 * 60


def monte_carlo(currencies_params, sizes, log_params, lp_params, simul_params, nb_MCs, seed, q):

    currencies, init_swap_price_01, mu, sigma = currencies_params
    nb_sizes = sizes.shape[0]
    lambda_, a, b = log_params
    intensity_functions_01_object = [Logistic(lambda_, a, b) for _ in range(nb_sizes)]
    intensity_functions_10_object = [Logistic(lambda_, a, b) for _ in range(nb_sizes)]
    market = Market(currencies, init_swap_price_01, mu, sigma, sizes, intensity_functions_01_object,
                    intensity_functions_10_object)
    typo, initial_inventories, initial_cash, extra_params = lp_params
    dt_sim, t_sim = simul_params

    lp = None
    if typo == 'POcst':
        delta = extra_params
        lp = LiquidityProviderCstDelta(
            'POcst%d' % delta, initial_inventories.copy(), initial_cash, market,
            PerfectOracle(), delta * 1e-4,
        )
    elif typo == 'LOcst':
        delta = extra_params
        lp = LiquidityProviderCstDelta(
            'LOcst%d' % delta, initial_inventories.copy(), initial_cash, market,
            LaggedOracle(dt_sim), delta * 1e-4,
        )
    elif typo == 'SOcst':
        delta = extra_params
        lp = LiquidityProviderCstDelta(
            'SOcst%d' % delta, initial_inventories.copy(), initial_cash, market,
            SparseOracle(dt_sim), delta * 1e-4,
        )
    elif typo == 'ammsqrt':
        delta = extra_params
        lp = LiquidityProviderAMMSqrt(
            'cfmm%d' % delta, initial_inventories.copy(), initial_cash, market,
            BaseOracle(), delta * 1e-4,
        )
    elif typo == 'swaapv1':
        name = extra_params['name']
        delta_in_bps = extra_params["delta_in_bps"] * 1e-4
        delta = delta_in_bps * 1e-4
        z = extra_params["z"]
        horizon = extra_params["horizon_in_dt"] * dt_sim
        lookback_calls = extra_params["lookback_calls"]
        lookback_step = extra_params["lookback_step"]
        lp = LiquidityProviderSwaapV1(
            f'mmm_{name}', initial_inventories.copy(), initial_cash, market,
            PerfectOracle(), delta,
            z, horizon, lookback_calls, lookback_step,
        )
    elif typo == 'POmyopic':
        delta = extra_params
        lp = LiquidityProviderCstDelta(
            'myopic', initial_inventories.copy(), initial_cash, market,
            PerfectOracle(), delta * 1e-4,
        )
    elif typo == 'PObcf':
        gamma = extra_params
        lp = LiquidityProviderBestClosedForm(
            'PObcf%.0e' % gamma, initial_inventories.copy(), initial_cash, market,
            PerfectOracle(), gamma,
        )
    elif typo == 'LObcf':
        gamma = extra_params
        lp = LiquidityProviderBestClosedForm(
            'LObcf%.0e' % gamma, initial_inventories.copy(), initial_cash, market,
            LaggedOracle(10./NUM_SEC_PER_DAY), gamma,
        )

    if lp is None:
        raise ValueError("Unrecognized LP")

    print('Starting ' + lp.name)

    pnls = np.zeros(nb_MCs)
    volumes = np.zeros(nb_MCs)
    np.random.seed(seed)

    for j in range(nb_MCs):
        if j % 10 == 0:
            print(lp.name + ': ' + str(j))
        lp.reset()
        res = market.simulate(dt_sim, t_sim, lp)
        pnls[j] = res.pnl[-1]
        volumes[j] = np.sum(res.volumes)

    res = (lp.name, np.mean(pnls), np.std(pnls), np.mean(volumes))
    print(f"name: {res[0]}, mean={res[1]}, std={res[2]}, volume={res[3]}")

    q.put(res)
    print('Done with ' + lp.name)


def main():
    currencies = ['BTC', 'ETH']
    init_swap_price_01 = 3000. / 40000.
    mu = 0.
    scale = 1. / 252.
    sigma = 0.2 * np.sqrt(scale)

    currencies_params = (currencies, init_swap_price_01, mu, sigma)

    sizes = np.array([1. / 40.])

    lambda_ = 45.
    a = -1.8
    b = 1300.

    log_params = lambda_, a, b

    initial_inventories = 20. * np.array([1., 1 / init_swap_price_01])
    initial_cash = 0.

    dt_sim = 10. / NUM_SEC_PER_DAY
    t_sim = 1.
    simul_params = (dt_sim, t_sim)

    nb_MCs = 100
    seed = 42

    q = Queue()

    names = []
    means = []
    stdevs = []
    volumes = []
    colors = []

    jobs = []
    typo = 'POcst'
    deltas = [5, 10, 30, 50]
    for delta in deltas:
        lp_params = (typo, initial_inventories, initial_cash, delta)
        job = Process(target=monte_carlo, args=(currencies_params, sizes, log_params, lp_params, simul_params, nb_MCs, seed, q))
        job.start()
        jobs.append(job)

    for job in jobs:
        job.join()

    for delta in deltas:
        name, mean, stdev, volume = q.get()
        names.append(name)
        means.append(mean)
        stdevs.append(stdev)
        volumes.append(volume)
        colors.append('blue')

    # jobs = []
    # typo = 'LOcst'
    # deltas = np.arange(1, 30, 5)
    # for delta in deltas:
    #     lp_params = (typo, initial_inventories, initial_cash, delta)
    #     job = Process(target=monte_carlo,
    #                   args=(currencies_params, sizes, log_params, lp_params, simul_params, nb_MCs, seed, q))
    #     job.start()
    #     jobs.append(job)
    #
    # for job in jobs:
    #     job.join()
    #
    # for delta in deltas:
    #     name, mean, stdev, volume = q.get()
    #     names.append(name)
    #     means.append(mean)
    #     stdevs.append(stdev)
    #     volumes.append(volume)
    #     colors.append('cyan')
    #
    # jobs = []
    # typo = 'SOcst'
    # deltas = np.arange(1, 30, 10)
    # for delta in deltas:
    #     lp_params = (typo, initial_inventories, initial_cash, delta)
    #     job = Process(target=monte_carlo,
    #                   args=(currencies_params, sizes, log_params, lp_params, simul_params, nb_MCs, seed, q))
    #     job.start()
    #     jobs.append(job)
    #
    # for job in jobs:
    #     job.join()
    #
    # for delta in deltas:
    #     name, mean, stdev, volume = q.get()
    #     names.append(name)
    #     means.append(mean)
    #     stdevs.append(stdev)
    #     volumes.append(volume)
    #     colors.append('cyan')

    # jobs = []
    # typo = 'POmyopic'
    # lt = LogisticTools(lambda_, a, b)
    # mq = lt.delta(0.) * 1e4
    # print(mq)
    #
    # lp_params = (typo, initial_inventories, initial_cash, mq)
    # job = Process(target=monte_carlo,
    #               args=(currencies_params, sizes, log_params, lp_params, simul_params, nb_MCs, seed, q))
    # job.start()
    # jobs.append(job)
    #
    # for job in jobs:
    #     job.join()
    #
    # name, mean, stdev, volume = q.get()
    # names.append(name)
    # means.append(mean)
    # stdevs.append(stdev)
    # volumes.append(volume)
    # colors.append('black')

    jobs = []
    typo = 'swaapv1'

    param_schema = [
        "delta_in_bps",
        "z",
        "horizon_in_dt",
        "lookback_calls",
        "lookback_step",
    ]
    param_values = [
        (2.5, 6, 1, 5, 3),
        (2.5, 6, 1.5, 5, 3),
        (2.5, 6, 2.5, 5, 3),
        (2.5, 6, 5, 5, 3),
    ]
    params = [
        dict((k, v) for k, v in zip(param_schema, param))
        for param in param_values
    ]
    for idx, param in enumerate(params):
        param["name"] = f'{idx}'
        lp_params = (typo, initial_inventories, initial_cash, param)
        job = Process(target=monte_carlo,
                      args=(currencies_params, sizes, log_params, lp_params, simul_params, nb_MCs, seed, q))
        job.start()
        jobs.append(job)

    for job in jobs:
        job.join()

    for param in params:
        name, mean, stdev, volume = q.get()
        names.append(name)
        means.append(mean)
        stdevs.append(stdev)
        volumes.append(volume)
        colors.append('purple')

    jobs = []
    typo = 'ammsqrt'
    deltas = [5, 30, 100]
    for delta in deltas:
        lp_params = (typo, initial_inventories, initial_cash, delta)
        job = Process(target=monte_carlo,
                      args=(currencies_params, sizes, log_params, lp_params, simul_params, nb_MCs, seed, q))
        job.start()
        jobs.append(job)

    for job in jobs:
        job.join()

    for delta in deltas:
        name, mean, stdev, volume = q.get()
        names.append(name)
        means.append(mean)
        stdevs.append(stdev)
        volumes.append(volume)
        colors.append('red')

    jobs = []
    typo = 'PObcf'
    gammas = [0., 10, 100., 5000., 100000., 500000.]
    for gamma in gammas:
        lp_params = (typo, initial_inventories, initial_cash, gamma)
        job = Process(target=monte_carlo,
                      args=(currencies_params, sizes, log_params, lp_params, simul_params, nb_MCs, seed, q))
        job.start()
        jobs.append(job)

    for job in jobs:
        job.join()

    for gamma in gammas:
        name, mean, stdev, volume = q.get()
        names.append(name)
        means.append(mean)
        stdevs.append(stdev)
        volumes.append(volume)
        colors.append('green')

    # jobs = []
    # typo = 'LObcf'
    # for gamma in gammas:
    #     lp_params = (typo, initial_inventories, initial_cash, gamma)
    #     job = Process(target=monte_carlo,
    #                   args=(currencies_params, sizes, log_params, lp_params, simul_params, nb_MCs, seed, q))
    #     job.start()
    #     jobs.append(job)
    #
    # for job in jobs:
    #     job.join()
    #
    # for gamma in gammas:
    #     name, mean, stdev, volume = q.get()
    #     names.append(name)
    #     means.append(mean)
    #     stdevs.append(stdev)
    #     volumes.append(volume)
    #     colors.append('orange')

    plt.rcParams["figure.figsize"] = [16, 9]
    fig, ax = plt.subplots(1, 1)

    ax.scatter(np.array(stdevs), np.array(means), c=colors, alpha=0.5)
    ax.set_xlabel('Standard deviation of PnL - PnL Hodl (in %s) after %.1f day(s)' % (currencies[0], t_sim))
    ax.set_ylabel('Mean of PnL - PnL Hodl (in %s) after %.1f day(s)' % (currencies[0], t_sim))
    ax.set_title('Statistics of PnL - PnL Hodl (in %s) after %.1f day(s) for different LP strategies' % (currencies[0], t_sim))

    for name, mean, stdev in zip(names, means, stdevs):
        ax.annotate(name, (stdev, mean))

    plt.savefig('frontier_basics.pdf')
    plt.show()


if __name__ == '__main__':
    main()