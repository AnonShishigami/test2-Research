from multiprocessing import Process, Queue
import warnings
import time

import matplotlib.pyplot as plt
import numpy as np

from ammlib import Logistic, MixedLogisticsExtended, Market,\
    BaseOracle, PerfectOracle, LaggedOracle, SparseOracle,\
    LiquidityProviderCstDelta, LiquidityProviderCFMMSqrt, LiquidityProviderCFMMSqrtCloseArb,\
    LiquidityProviderBestClosedForm,\
    LiquidityProviderSwaapV1,\
    LiquidityProviderCurveV2,\
    LiquidityProviderConcentratedCFMMSqrt


warnings.filterwarnings("ignore")

NUM_SEC_PER_DAY = 24 * 60 * 60
BPS_PRECISION = 1e-4


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

    lp_init = None
    if typo == 'POcst':
        def lp_init():
            delta = extra_params
            lp = LiquidityProviderCstDelta(
                'POcst%d' % delta, initial_inventories.copy(), initial_cash, market,
                PerfectOracle(), True, delta * BPS_PRECISION,
            )
            return lp
    elif typo == 'POcst_noarb':
        def lp_init():
            delta = extra_params
            lp = LiquidityProviderCstDelta(
                'POcst_noarb%d' % delta, initial_inventories.copy(), initial_cash, market,
                PerfectOracle(), False, delta * BPS_PRECISION,
            )
            return lp
    elif typo == 'LOcst':
        def lp_init():
            delta = extra_params
            lp = LiquidityProviderCstDelta(
                'LOcst%d' % delta, initial_inventories.copy(), initial_cash, market,
                LaggedOracle(dt_sim), True, delta * BPS_PRECISION,
            )
            return lp
    elif typo == 'LOcst_noarb':
        def lp_init():
            delta = extra_params
            lp = LiquidityProviderCstDelta(
                'LOcst_noarb%d' % delta, initial_inventories.copy(), initial_cash, market,
                LaggedOracle(dt_sim), False, delta * BPS_PRECISION,
            )
            return lp
    elif typo == 'SOcst':
        def lp_init():
            delta = extra_params
            lp = LiquidityProviderCstDelta(
                'SOcst%d' % delta, initial_inventories.copy(), initial_cash, market,
                SparseOracle(dt_sim), True, delta * BPS_PRECISION,
            )
            return lp
    elif typo == 'SOcst_noarb':
        def lp_init():
            delta = extra_params
            lp = LiquidityProviderCstDelta(
                'SOcst_noarb%d' % delta, initial_inventories.copy(), initial_cash, market,
                SparseOracle(dt_sim), False, delta * BPS_PRECISION,
            )
            return lp
    elif typo == 'cfmmsqrt':
        def lp_init():
            delta = extra_params
            lp = LiquidityProviderCFMMSqrt(
                'cfmmsqrt%d' % delta, initial_inventories.copy(), initial_cash, market,
                BaseOracle(), True, delta * BPS_PRECISION,
            )
            return lp
    elif typo == 'cfmmsqrt_noarb':
        def lp_init():
            delta = extra_params
            lp = LiquidityProviderCFMMSqrt(
                'cfmmsqrt_noarb%d' % delta, initial_inventories.copy(), initial_cash, market,
                BaseOracle(), False, delta * BPS_PRECISION,
            )
            return lp
    elif typo == 'cfmmsqrt_closearb':
        def lp_init():
            delta = extra_params
            lp = LiquidityProviderCFMMSqrtCloseArb(
                'cfmmsqrt_closearb%d' % delta, initial_inventories.copy(), initial_cash, market,
                BaseOracle(), True, delta * BPS_PRECISION,
            )
            return lp
    elif typo == 'conc_cfmmsqrt':
        def lp_init():
            delta = extra_params
            lp = LiquidityProviderConcentratedCFMMSqrt(
                'conc_cfmmsqrt%d' % delta, initial_inventories.copy(), initial_cash, market,
                BaseOracle(), True, delta * BPS_PRECISION,
            )
            return lp
    elif typo == 'conc_cfmmsqrt_noarb':
        def lp_init():
            delta = extra_params
            lp = LiquidityProviderConcentratedCFMMSqrt(
                'conc_cfmmsqrt_noarb%d' % delta, initial_inventories.copy(), initial_cash, market,
                BaseOracle(), False, delta * BPS_PRECISION,
            )
            return lp
    elif typo == 'swaapv1':
        def lp_init():
            name = extra_params['name']
            delta_in_bps = extra_params["delta_in_bps"] * BPS_PRECISION
            delta = delta_in_bps * BPS_PRECISION
            z = extra_params["z"]
            horizon = extra_params["horizon_in_dt"] * dt_sim
            lookback_calls = extra_params["lookback_calls"]
            lookback_step = extra_params["lookback_step"]
            lp = LiquidityProviderSwaapV1(
                f'mmm_{name}', initial_inventories.copy(), initial_cash, market,
                SparseOracle(dt_sim), True, delta,
                z, horizon, lookback_calls, lookback_step,
            )
            return lp
    elif typo == 'swaapv1_noarb':
        def lp_init():
            name = extra_params['name']
            delta_in_bps = extra_params["delta_in_bps"] * BPS_PRECISION
            delta = delta_in_bps * BPS_PRECISION
            z = extra_params["z"]
            horizon = extra_params["horizon_in_dt"] * dt_sim
            lookback_calls = extra_params["lookback_calls"]
            lookback_step = extra_params["lookback_step"]
            lp = LiquidityProviderSwaapV1(
                f'mmm_{name}_noarb', initial_inventories.copy(), initial_cash, market,
                PerfectOracle(), False, delta,
                z, horizon, lookback_calls, lookback_step,
            )
            return lp
    elif typo == "curvev2":
        def lp_init():
            name = extra_params['name']
            initial_prices = extra_params["initial_prices"]
            lp = LiquidityProviderCurveV2(
                f'curve_v2_{name}', initial_inventories.copy(), initial_cash, market,
                PerfectOracle(), True, initial_prices
            )
            return lp
    elif typo == "curvev2_noarb":
        def lp_init():
            name = extra_params['name']
            initial_prices = extra_params["initial_prices"]
            lp = LiquidityProviderCurveV2(
                f'curve_v2_{name}_noarb', initial_inventories.copy(), initial_cash, market,
                PerfectOracle(), False, initial_prices
            )
            return lp
    elif typo == 'POmyopic':
        def lp_init():
            delta = extra_params
            lp = LiquidityProviderCstDelta(
                'myopic', initial_inventories.copy(), initial_cash, market,
                PerfectOracle(), True, delta * BPS_PRECISION,
            )
            return lp
    elif typo == 'POmyopic_noarb':
        def lp_init():
            delta = extra_params
            lp = LiquidityProviderCstDelta(
                'myopic', initial_inventories.copy(), initial_cash, market,
                PerfectOracle(), False, delta * BPS_PRECISION,
            )
            return lp
    elif typo == 'PObcf':
        def lp_init():
            gamma = extra_params
            lp = LiquidityProviderBestClosedForm(
                'PObcf%.0e' % gamma, initial_inventories.copy(), initial_cash, market,
                PerfectOracle(), True, gamma,
            )
            return lp
    elif typo == 'PObcf_noarb':
        def lp_init():
            gamma = extra_params
            lp = LiquidityProviderBestClosedForm(
                'PObcf_noarb%.0e' % gamma, initial_inventories.copy(), initial_cash, market,
                PerfectOracle(), False, gamma,
            )
            return lp
    elif typo == 'SObcf':
        def lp_init():
            gamma = extra_params
            lp = LiquidityProviderBestClosedForm(
                'SObcf%.0e' % gamma, initial_inventories.copy(), initial_cash, market,
                SparseOracle(10./NUM_SEC_PER_DAY), True, gamma,
            )
            return lp
    elif typo == 'SObcf_noarb':
        def lp_init():
            gamma = extra_params
            lp = LiquidityProviderBestClosedForm(
                'PObcf_noarb%.0e' % gamma, initial_inventories.copy(), initial_cash, market,
                LaggedOracle(10./NUM_SEC_PER_DAY), False, gamma,
            )
            return lp

    if lp_init is None:
        raise ValueError("Unrecognized LP:", typo)

    lp_name = lp_init().name
    print('Starting ' + lp_name)

    pnls = np.zeros(nb_MCs)
    volumes = np.zeros(nb_MCs)
    np.random.seed(seed)

    for j in range(nb_MCs):
        lp = lp_init()
        if j % 10 == 0:
            print(lp_name + ': ' + str(j))
        res = market.simulate(dt_sim, t_sim, lp)
        pnls[j] = res.pnl[-1]
        volumes[j] = np.sum(res.volumes)

    res = (lp_name, np.mean(pnls), np.std(pnls), np.mean(volumes))
    print(f"name: {res[0]}, mean={res[1]}, std={res[2]}, volume={res[3]}")

    q.put(res)
    print('Done with ' + lp_name)


def main():
    currencies = ['BTC', 'ETH']
    initial_prices = [40000., 3000.]
    init_swap_price_01 = initial_prices[1] / initial_prices[0]
    scale = 1. / 365.
    mu = 0. * scale
    sigma = 0.6 * np.sqrt(scale)
    print(f"mu={mu}, sigma={sigma}")

    currencies_params = (currencies, init_swap_price_01, mu, sigma)

    sizes = np.array([1. / 300.])

    lambda_ = 700.
    a = -1.8
    b = 1300.

    print(f"sizes={sizes}, lambda_={lambda_}, a={a}, b={b}")

    log_params = lambda_, a, b

    initial_inventories = 20. * np.array([1., 1 / init_swap_price_01])
    initial_cash = 0.

    dt_sim = 10. / NUM_SEC_PER_DAY
    t_sim = 1.
    simul_params = (dt_sim, t_sim)

    print(f"dt_sim={dt_sim}, t_sim={t_sim}")

    nb_MCs = 10
    seed = 42

    q = Queue()

    names = []
    means = []
    stdevs = []
    volumes = []
    colors = []

    for typo in [
        "POcst_noarb",
        "swaapv1",
        "swaapv1_noarb",
        "conc_cfmmsqrt",
        "cfmmsqrt",
        "cfmmsqrt_closearb",
        "PObcf_noarb",
        "SObcf",
        "POmyopic_noarb",
        "curvev2_noarb",
        "curvev2",
    ]:

        start = time.time()
        if "POcst" in typo:
            if typo == "POcst":
                color = "chartreuse"
            elif typo == "POcst_noarb":
                color = "blue"
            else:
                raise ValueError("Unrecognized typo:", typo)
            jobs = []
            deltas = [5, 15, 30]
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
                colors.append(color)

        elif "LOcst" in typo:
            if typo == "LOcst":
                color = "cyan"
            elif typo == "LOcst_noarb":
                color = "lime"
            else:
                raise ValueError("Unrecognized typo:", typo)
            jobs = []
            deltas = [5, 15, 30]
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
                colors.append(color)

        elif "SOcst" in typo:
            if typo == "SOcst":
                color = "fuchsia"
            elif typo == "SOcst_noarb":
                color = "lavender"
            else:
                raise ValueError("Unrecognized typo:", typo)
            jobs = []
            deltas = [5, 15, 30]
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
                colors.append(color)

        elif typo == "POmyopic_noarb":
            ext = MixedLogisticsExtended(Logistic(lambda_, a, b), Logistic(lambda_/2., 0., 10.*b))
            mq = ext.delta0 * 1e4
            print("myopic quote:", mq)
            lp_params = (typo, initial_inventories, initial_cash, mq)
            job = Process(target=monte_carlo,
                          args=(currencies_params, sizes, log_params, lp_params, simul_params, nb_MCs, seed, q))
            job.start()
            job.join()
            name, mean, stdev, volume = q.get()
            names.append(name)
            means.append(mean)
            stdevs.append(stdev)
            volumes.append(volume)
            colors.append("black")

        elif "curvev2" in typo:
            if typo == "curvev2":
                color = "lightcoral"
            elif typo == "curvev2_noarb":
                color = "crimson"
            else:
                raise ValueError("Unrecognized typo:", typo)
            param = {
                "name": "tricrypto",
                "initial_prices": initial_prices
            }
            lp_params = (typo, initial_inventories, initial_cash, param)
            job = Process(target=monte_carlo,
                            args=(currencies_params, sizes, log_params, lp_params, simul_params, nb_MCs, seed, q))
            job.start()
            job.join()
            name, mean, stdev, volume = q.get()
            names.append(name)
            means.append(mean)
            stdevs.append(stdev)
            volumes.append(volume)
            colors.append(color)

        elif "swaapv1" in typo:
            if typo == "swaapv1":
                color = "purple"
            elif typo == "swaapv1_noarb":
                color = "brown"
            else:
                raise ValueError("Unrecognized typo:", typo)
            param_values = [
                # (0, 6, 0, 5, 4),  # no spread
                # (2.5, 6, 1, 5, 4),
                # (2.5, 6, 1.5, 5, 4),
                (2.5, 6, 2.5, 5, 4),
                (2.5, 6, 5, 5, 4),
                (5, 6, 2.5, 5, 4),
                # (5, 4, 2.5, 5, 4),
                # (5, 4, 5, 5, 4),
                # (10, 2, 2.5, 5, 4),
                # (10, 2, 5, 5, 4),
                # (20, 0.6, 2.5, 5, 4),
                # (20, 1, 2.5, 5, 4),
            ]
            param_schema = [
                "delta_in_bps",
                "z",
                "horizon_in_dt",
                "lookback_calls",
                "lookback_step",
            ]
            jobs = []
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
                colors.append(color)

        elif "cfmmsqrt" in typo:
            if typo == "cfmmsqrt_closearb":
                color = "red"
            elif typo == "cfmmsqrt":
                color = "gray"
            elif typo == "cfmmsqrt_noarb":
                color = "olive"
            elif typo == "conc_cfmmsqrt":
                color = "gray"
            elif typo == "conc_cfmmsqrt_noarb":
                color = "olive"
            else:
                raise ValueError("Unrecognized typo:", typo)
            deltas = [1, 5, 30, 100]
            jobs = []
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
                colors.append(color)
            
        elif "PObcf" in typo:
            if typo == "PObcf":
                color = "green"
            elif typo == "PObcf_noarb":
                color = "pink"
            else:
                raise ValueError("Unrecognized typo:", typo)
            gammas = [0., 10, 100., 5000., 10000, 50000]
            jobs = []
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
                colors.append("green")

        elif "SObcf" in typo:
            if typo == "SObcf":
                color = "orange"
            elif typo == "SObcf_noarb":
                color = "yellow"
            else:
                raise ValueError("Unrecognized typo:", typo)
            gammas = [0., 10, 100., 5000., 10000, 50000]
            jobs = []
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
                colors.append(color)
                
        end = time.time()
        print(f"{typo}: time={end - start}s")

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
