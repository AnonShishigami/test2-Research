from multiprocessing import Process, Queue
import warnings
import time

import matplotlib.pyplot as plt
import numpy as np

# tools
from sandbox.demand_curve import Logistic
from sandbox.control_tools import MixedLogisticsExtended
from sandbox.market import Market
from sandbox.oracle import BaseOracle, PerfectOracle, LaggedOracle, SparseOracle
# strategies
from sandbox.strategies.swaap_v1.strategy import SwaapV1
from sandbox.strategies.cfmm_sqrt.strategy import CFMMSqrt
from sandbox.strategies.cst_delta.strategy import CstDelta
from sandbox.strategies.curve_v2.strategy import CurveV2
from sandbox.strategies.best_closed_form.strategy import BestClosedForm


warnings.filterwarnings("ignore")

NUM_TRADING_DAYS_PER_YEAR = 365.242199
NUM_SECS_PER_DAY = 24 * 60 * 60
BPS_PRECISION = 1e-4
RFR = 0. / 100
DT_ORACLE = 12 / NUM_SECS_PER_DAY


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
            lp = CstDelta(
                'POcst%d' % delta, initial_inventories.copy(), initial_cash, market,
                PerfectOracle(), True, delta * BPS_PRECISION,
            )
            return lp
    elif typo == 'POcst_noarb':
        def lp_init():
            delta = extra_params
            lp = CstDelta(
                'POcst_noarb%d' % delta, initial_inventories.copy(), initial_cash, market,
                PerfectOracle(), False, delta * BPS_PRECISION,
            )
            return lp
    elif typo == 'LOcst':
        def lp_init():
            delta = extra_params
            lp = CstDelta(
                'LOcst%d' % delta, initial_inventories.copy(), initial_cash, market,
                LaggedOracle(DT_ORACLE), True, delta * BPS_PRECISION,
            )
            return lp
    elif typo == 'LOcst_noarb':
        def lp_init():
            delta = extra_params
            lp = CstDelta(
                'LOcst_noarb%d' % delta, initial_inventories.copy(), initial_cash, market,
                LaggedOracle(DT_ORACLE), False, delta * BPS_PRECISION,
            )
            return lp
    elif typo == 'SOcst':
        def lp_init():
            delta = extra_params
            lp = CstDelta(
                'SOcst%d' % delta, initial_inventories.copy(), initial_cash, market,
                SparseOracle(DT_ORACLE), True, delta * BPS_PRECISION,
            )
            return lp
    elif typo == 'SOcst_noarb':
        def lp_init():
            delta = extra_params
            lp = CstDelta(
                'SOcst_noarb%d' % delta, initial_inventories.copy(), initial_cash, market,
                SparseOracle(DT_ORACLE), False, delta * BPS_PRECISION,
            )
            return lp
    elif typo == 'cfmmsqrt':
        def lp_init():
            delta = extra_params
            lp = CFMMSqrt(
                'cfmmsqrt%d' % delta, initial_inventories.copy(), initial_cash, market,
                BaseOracle(), True, delta * BPS_PRECISION,
            )
            return lp
    elif typo == 'cfmmsqrt_noarb':
        def lp_init():
            delta = extra_params
            lp = CFMMSqrt(
                'cfmmsqrt_noarb%d' % delta, initial_inventories.copy(), initial_cash, market,
                BaseOracle(), False, delta * BPS_PRECISION,
            )
            return lp
    elif typo == 'swaapv1':
        def lp_init():
            name = extra_params['name']
            delta_in_bps = extra_params["delta_in_bps"]
            delta = delta_in_bps * BPS_PRECISION
            z = extra_params["z"]
            horizon = extra_params["horizon_in_dt"] / NUM_SECS_PER_DAY
            lookback_calls = extra_params["lookback_calls"]
            lookback_step = extra_params["lookback_step"]
            lp = SwaapV1(
                f'mmm_{name}', initial_inventories.copy(), initial_cash, market,
                SparseOracle(DT_ORACLE), True, delta,
                z, horizon, lookback_calls, lookback_step,
            )
            return lp
    elif typo == 'swaapv1_noarb':
        def lp_init():
            name = extra_params['name']
            delta_in_bps = extra_params["delta_in_bps"] * BPS_PRECISION
            delta = delta_in_bps * BPS_PRECISION
            z = extra_params["z"]
            horizon = extra_params["horizon_in_dt"] / NUM_SECS_PER_DAY
            lookback_calls = extra_params["lookback_calls"]
            lookback_step = extra_params["lookback_step"]
            lp = SwaapV1(
                f'mmm_{name}_noarb', initial_inventories.copy(), initial_cash, market,
                PerfectOracle(), False, delta,
                z, horizon, lookback_calls, lookback_step,
            )
            return lp
    elif typo == "curvev2_tricrypto":
        def lp_init():
            name = extra_params['name']
            initial_prices = extra_params["initial_prices"]
            A = extra_params["A"]
            gamma = extra_params["gamma"]
            lp = CurveV2(
                f'{name}', initial_inventories.copy(), initial_cash, market,
                PerfectOracle(), True, initial_prices, dt_sim, 
                A=A, gamma=gamma
            )
            return lp
    elif typo == "curvev2_tricrypto_noarb":
        def lp_init():
            name = extra_params['name']
            initial_prices = extra_params["initial_prices"]
            A = extra_params["A"]
            gamma = extra_params["gamma"]
            lp = CurveV2(
                f'{name}_noarb', initial_inventories.copy(), initial_cash, market,
                PerfectOracle(), False, initial_prices, dt_sim, 
                A=A, gamma=gamma
            )
            return lp
    elif typo == 'POmyopic':
        def lp_init():
            delta = extra_params
            lp = CstDelta(
                'myopic', initial_inventories.copy(), initial_cash, market,
                PerfectOracle(), True, delta * BPS_PRECISION,
            )
            return lp
    elif typo == 'POmyopic_noarb':
        def lp_init():
            delta = extra_params
            lp = CstDelta(
                'myopic', initial_inventories.copy(), initial_cash, market,
                PerfectOracle(), False, delta * BPS_PRECISION,
            )
            return lp
    elif typo == 'PObcf':
        def lp_init():
            gamma = extra_params
            lp = BestClosedForm(
                'PObcf%.0e' % gamma, initial_inventories.copy(), initial_cash, market,
                PerfectOracle(), True, gamma,
            )
            return lp
    elif typo == 'PObcf_noarb':
        def lp_init():
            gamma = extra_params
            lp = BestClosedForm(
                'PObcf_noarb%.0e' % gamma, initial_inventories.copy(), initial_cash, market,
                PerfectOracle(), False, gamma,
            )
            return lp
    elif typo == 'SObcf':
        def lp_init():
            gamma = extra_params
            lp = BestClosedForm(
                'SObcf%.0e' % gamma, initial_inventories.copy(), initial_cash, market,
                SparseOracle(DT_ORACLE), True, gamma,
            )
            return lp
    elif typo == 'SObcf_noarb':
        def lp_init():
            gamma = extra_params
            lp = BestClosedForm(
                'PObcf_noarb%.0e' % gamma, initial_inventories.copy(), initial_cash, market,
                LaggedOracle(DT_ORACLE), False, gamma,
            )
            return lp

    if lp_init is None:
        raise ValueError("Unrecognized LP:", typo)

    lp_name = lp_init().name
    print('Starting ' + lp_name)

    pnls = np.zeros(nb_MCs)
    volumes = np.zeros(nb_MCs)
    arb_volumes = np.zeros(nb_MCs)
    proposed_swap_price_diffs = []
    np.random.seed(seed)

    for j in range(nb_MCs):
        lp = lp_init()
        if j % 10 == 0:
            print(lp_name + ': ' + str(j))
        _res = market.simulate(dt_sim, t_sim, lp)
        pnls[j] = _res.pnl[-1]
        volumes[j] = np.sum(_res.volumes)
        arb_volumes = np.sum(_res.arb_volumes)
        proposed_swap_price_diffs += _res.proposed_swap_price_diffs.tolist()

    res = (lp_name, np.mean(pnls), np.std(pnls), np.mean(volumes), np.mean(arb_volumes))
    apr = res[1]  * NUM_TRADING_DAYS_PER_YEAR / t_sim
    apstd = res[2] * np.sqrt(NUM_TRADING_DAYS_PER_YEAR / t_sim)
    s = f"name: {res[0]}, apr={apr * 100:.4f}% mean={res[1] * 100:.6f}%, var={res[2] * 100:.6f}%, retail={res[3]:.2f}, arb={res[4]:.2f}, sharpe={res[1] / res[2]:.2f}"
    # s += '\nprice delta distribution:'
    # for per in [0, 1, 10, 25, 50, 75, 90, 99, 100]:
    #     s += f'\np{per}: {100 * np.percentile(proposed_swap_price_diffs, per)}%'
    print(s)

    q.put(res)
    print('Done with ' + lp_name)


def main():
    currencies = ['BTC', 'ETH']
    initial_prices = [40000., 3000.]
    init_swap_price_01 = initial_prices[1] / initial_prices[0]
    initial_inventories = 400. * np.array([1., 1 / init_swap_price_01])

    scale = 1. / NUM_TRADING_DAYS_PER_YEAR
    mu = 0 * scale
    sigma = 0.8 * np.sqrt(scale)
    print(f"mu={mu}, sigma={sigma}")

    dt_norm_factor = 1. / NUM_SECS_PER_DAY
    dt_step = 2
    dt_sim = dt_step * dt_norm_factor
    assert dt_sim <= DT_ORACLE  # TODO: remove this
    assert (DT_ORACLE % dt_sim) < 1e-15  # TODO: remove this
    t_sim = 1
    simul_params = (dt_sim, t_sim)

    currencies_params = (currencies, init_swap_price_01, mu, sigma)

    sizes = np.array([initial_inventories[0] * 2 / 1000.])

    lambda_ = 750
    a = -1.8
    b = 1300

    print(f"sizes={sizes}, lambda_={lambda_}, a={a}, b={b}")

    log_params = lambda_, a, b

    initial_cash = 0.

    print(f"dt_sim={dt_sim}, t_sim={t_sim}")

    nb_MCs = 100
    seed = 42

    q = Queue()

    names = []
    means = []
    stdevs = []
    volumes = []
    arb_volumes = []
    colors = []

    for typo in [
        "POcst_noarb",
        "swaapv1",
        "swaapv1_noarb",
        "cfmmsqrt",
        "cfmmsqrt_noarb",
        "PObcf_noarb",
        "SObcf",
        "POmyopic_noarb",
        "curvev2_tricrypto_noarb",
        "curvev2_tricrypto",
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
                name, mean, stdev, volume, arb_volume = q.get()
                names.append(name)
                means.append(mean)
                stdevs.append(stdev)
                volumes.append(volume)
                arb_volumes.append(arb_volume)
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
                name, mean, stdev, volume, arb_volume = q.get()
                names.append(name)
                means.append(mean)
                stdevs.append(stdev)
                volumes.append(volume)
                arb_volumes.append(arb_volume)
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
                name, mean, stdev, volume, arb_volume = q.get()
                names.append(name)
                means.append(mean)
                stdevs.append(stdev)
                volumes.append(volume)
                arb_volumes.append(arb_volume)
                colors.append(color)

        elif typo == "POmyopic_noarb":
            ext = MixedLogisticsExtended(Logistic(lambda_, a, b), Logistic(lambda_ / 2., 0., 10. * b))
            mq = ext.delta0 * 1e4
            print("myopic quote:", mq)
            lp_params = (typo, initial_inventories, initial_cash, mq)
            job = Process(target=monte_carlo,
                          args=(currencies_params, sizes, log_params, lp_params, simul_params, nb_MCs, seed, q))
            job.start()
            job.join()
            name, mean, stdev, volume, arb_volume = q.get()
            names.append(name)
            means.append(mean)
            stdevs.append(stdev)
            volumes.append(volume)
            arb_volumes.append(arb_volume)
            colors.append("black")

        elif "curvev2_tricrypto" in typo:
            if typo == "curvev2_tricrypto":
                color = "lightcoral"
            elif typo == "curvev2_tricrypto_noarb":
                color = "crimson"
            else:
                raise ValueError("Unrecognized typo:", typo)
            param_values = [
                (5400000, 20000000000000, "polygon"),
                (1707629, 11809167828997, "ethereum"),
            ]
            param_schema = [
                "A",
                "gamma",
                "chain"
            ]
            jobs = []
            params = [
                dict((k, v) for k, v in zip(param_schema, param))
                for param in param_values
            ]
            for idx, param in enumerate(params):
                param["name"] = f'tricrypto_{param["chain"]}'
                param["initial_prices"] = initial_prices
                lp_params = (typo, initial_inventories, initial_cash, param)
                job = Process(target=monte_carlo,
                              args=(currencies_params, sizes, log_params, lp_params, simul_params, nb_MCs, seed, q))
                job.start()
                jobs.append(job)

            lp_params = (typo, initial_inventories, initial_cash, param)
            job = Process(target=monte_carlo,
                            args=(currencies_params, sizes, log_params, lp_params, simul_params, nb_MCs, seed, q))
            for job in jobs:
                job.join()
            for param in params:
                name, mean, stdev, volume, arb_volume = q.get()
                names.append(name)
                means.append(mean)
                stdevs.append(stdev)
                volumes.append(volume)
                arb_volumes.append(arb_volume)
                colors.append(color)

        elif "swaapv1" in typo:
            if typo == "swaapv1":
                color = "purple"
            elif typo == "swaapv1_noarb":
                color = "brown"
            else:
                raise ValueError("Unrecognized typo:", typo)
            param_values = [
                (0, 6, 0, 1, 1),  # no spread, no fees
                (15, 6, 0, 1, 1),  # no spread, only fees
                (2.5, 6, 1, 5, 4),
                (2.5, 4, 2.5, 5, 4),
                (2.5, 3, 5, 5, 4),
                (2.5, 6, 2.5, 5, 4),
                (2.5, 6, 5, 5, 4),
                (5, 6, 4, 5, 4),
                (5, 4, 5, 5, 4),
                (5, 3, 2.5, 5, 4),
                (7.5, 3, 2.5, 5, 4),
                (10, 2, 5, 5, 4),
                (15, 2, 5, 5, 4),
                (20, 0.6, 2.5, 5, 4),
                (20, 1, 2.5, 5, 4),
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
                name, mean, stdev, volume, arb_volume = q.get()
                names.append(name)
                means.append(mean)
                stdevs.append(stdev)
                volumes.append(volume)
                arb_volumes.append(arb_volume)
                colors.append(color)

        elif "cfmmsqrt" in typo:
            if typo == "cfmmsqrt":
                color = "gray"
            elif typo == "cfmmsqrt_noarb":
                color = "olive"
            else:
                raise ValueError("Unrecognized typo:", typo)
            deltas = [5, 30, 100]
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
                name, mean, stdev, volume, arb_volume = q.get()
                names.append(name)
                means.append(mean)
                stdevs.append(stdev)
                volumes.append(volume)
                arb_volumes.append(arb_volume)
                colors.append(color)
            
        elif "PObcf" in typo:
            if typo == "PObcf":
                color = "green"
            elif typo == "PObcf_noarb":
                color = "pink"
            else:
                raise ValueError("Unrecognized typo:", typo)
            gammas = [0., 100., 10000, 100000]
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
                name, mean, stdev, volume, arb_volume = q.get()
                names.append(name)
                means.append(mean)
                stdevs.append(stdev)
                volumes.append(volume)
                arb_volumes.append(arb_volume)
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
                name, mean, stdev, volume, arb_volume = q.get()
                names.append(name)
                means.append(mean)
                stdevs.append(stdev)
                volumes.append(volume)
                arb_volumes.append(arb_volume)
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
