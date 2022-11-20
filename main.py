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
from sandbox.strategies.cfmm_powers.strategy import CFMMPowers
from sandbox.strategies.cfmm_sqrt.strategy import CFMMSqrt
from sandbox.strategies.cst_delta.strategy import CstDelta
from sandbox.strategies.curve_v2.strategy import CurveV2
from sandbox.strategies.best_closed_form.strategy import BestClosedForm


warnings.filterwarnings("ignore")

NUM_TRADING_DAYS_PER_YEAR = 365.242199
NUM_SECS_PER_DAY = 24 * 60 * 60
BPS_PRECISION = 1e-4
RFR = 0. / 100
DT_ORACLE = 10 / NUM_SECS_PER_DAY


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
            delta, concentration = extra_params
            lp = CFMMSqrt(
                f'cfmmsqrt{delta}{"" if concentration == 1 else f"_conc{concentration}"}', initial_inventories.copy(), initial_cash, market,
                BaseOracle(), True, delta * BPS_PRECISION, concentration=concentration
            )
            return lp
    elif typo == 'cfmmsqrt_noarb':
        def lp_init():
            delta, concentration = extra_params
            lp = CFMMSqrt(
                f'cfmmsqrt{delta}{"" if concentration == 1 else f"_conc{concentration}"}', initial_inventories.copy(), initial_cash, market,
                BaseOracle(), False, delta * BPS_PRECISION, concentration=concentration
            )
            return lp
    elif typo == 'cfmmpowers':
        def lp_init():
            delta, weights = extra_params["delta"], extra_params["weights"]
            lp = CFMMPowers(
                f"cfmmpowers{delta}w{weights[0]:.2f}", initial_inventories.copy(), initial_cash, market,
                BaseOracle(), True, weights=weights, delta=delta * BPS_PRECISION, 
            )
            return lp
    elif typo == 'cfmmpowers_noarb':
        def lp_init():
            delta, weights = extra_params["delta"], extra_params["weights"]
            lp = CFMMPowers(
                f"cfmmpowers{delta}w{weights[0]:.2f}_noarb", initial_inventories.copy(), initial_cash, market,
                BaseOracle(), False, weights=weights, delta=delta * BPS_PRECISION,
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
            concentration = extra_params["concentration"]
            lp = SwaapV1(
                f'mmm_{name}', initial_inventories.copy(), initial_cash, market,
                SparseOracle(DT_ORACLE), True, delta,
                z, horizon, lookback_calls, lookback_step,
                concentration=concentration
            )
            return lp
    elif typo == 'swaapv1_noarb':
        def lp_init():
            name = extra_params['name']
            delta_in_bps = extra_params["delta_in_bps"]
            delta = delta_in_bps * BPS_PRECISION
            z = extra_params["z"]
            horizon = extra_params["horizon_in_dt"] / NUM_SECS_PER_DAY
            lookback_calls = extra_params["lookback_calls"]
            lookback_step = extra_params["lookback_step"]
            lp = SwaapV1(
                f'mmm_{name}_noarb', initial_inventories.copy(), initial_cash, market,
                SparseOracle(DT_ORACLE), False, delta,
                z, horizon, lookback_calls, lookback_step,
            )
            return lp
    elif typo == "curvev2":
        def lp_init():
            name = extra_params['name']
            initial_prices = extra_params["initial_prices"]
            A = extra_params["A"]
            gamma = extra_params["gamma"]
            adjustment_step = extra_params["adjustment_step"]
            mid_fee = extra_params["mid_fee"]
            out_fee = extra_params["out_fee"]
            allowed_extra_profit = extra_params["allowed_extra_profit"]
            fee_gamma = extra_params["fee_gamma"]
            admin_fee = extra_params["admin_fee"]
            ma_half_time = extra_params["ma_half_time"]
            lp = CurveV2(
                f'{name}', initial_inventories.copy(), initial_cash, market,
                PerfectOracle(), True, initial_prices, dt_sim, 
                A=A, gamma=gamma,
                mid_fee=mid_fee,
                out_fee=out_fee,
                allowed_extra_profit=allowed_extra_profit,
                fee_gamma=fee_gamma,
                adjustment_step=adjustment_step,
                admin_fee=admin_fee,
                ma_half_time=ma_half_time,
            )
            return lp
    elif typo == "curvev2_noarb":
        def lp_init():
            name = extra_params['name']
            initial_prices = extra_params["initial_prices"]
            A = extra_params["A"]
            gamma = extra_params["gamma"]
            adjustment_step = extra_params["adjustment_step"]
            mid_fee = extra_params["mid_fee"]
            out_fee = extra_params["out_fee"]
            allowed_extra_profit = extra_params["allowed_extra_profit"]
            fee_gamma = extra_params["fee_gamma"]
            admin_fee = extra_params["admin_fee"]
            ma_half_time = extra_params["ma_half_time"]
            lp = CurveV2(
                f'{name}_noarb', initial_inventories.copy(), initial_cash, market,
                PerfectOracle(), False, initial_prices, dt_sim, 
                A=A,
                gamma=gamma,
                mid_fee=mid_fee,
                out_fee=out_fee,
                allowed_extra_profit=allowed_extra_profit,
                fee_gamma=fee_gamma,
                adjustment_step=adjustment_step,
                admin_fee=admin_fee,
                ma_half_time=ma_half_time,
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
                'SObcf_noarb%.0e' % gamma, initial_inventories.copy(), initial_cash, market,
                SparseOracle(DT_ORACLE), False, gamma,
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
    currencies = ['USD', 'ETH']
    initial_prices = [1., 1600.]
    init_swap_price_01 = initial_prices[1] / initial_prices[0]
    initial_inventories = 1000000. * np.array([1., 1 / init_swap_price_01])

    scale = 1. / NUM_TRADING_DAYS_PER_YEAR
    mu = 0.5 * scale
    sigma = 0.8 * np.sqrt(scale)
    print(f"mu={mu}, sigma={sigma}")

    dt_norm_factor = 1. / NUM_SECS_PER_DAY
    dt_step = 2
    dt_sim = dt_step * dt_norm_factor
    assert dt_sim <= DT_ORACLE  # TODO: remove this
    assert (DT_ORACLE % dt_sim) < 1e-15  # TODO: remove this
    t_sim = 0.1
    simul_params = (dt_sim, t_sim)

    currencies_params = (currencies, init_swap_price_01, mu, sigma)

    sizes = np.array([initial_inventories[0] * 2 / 120000])

    lambda_ = 5000.
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
        "cfmmpowers",
        "cfmmsqrt_noarb",
        "cfmmsqrt",
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
            deltas = [1, 5, 10, 30]
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

        elif "curvev2" in typo:
            if typo == "curvev2":
                color = "lightcoral"
            elif typo == "curvev2_noarb":
                color = "crimson"
            else:
                raise ValueError("Unrecognized typo:", typo)
            param_values = [
                (5400000, 20000000000000, 5000000, 30000000, 200000000000, 500000000000000, 500000000000000, 5000000000, 600, "3crypto-polygon"),  # 3crypto; polygon
                (1707629, 11809167828997, 5000000, 30000000, 2000000000000, 500000000000000, 2000000000000000, 5000000000, 600, "3crypto-ethereum"),  # 3crypto; ethereum
                (540000, 2000000000000, 5000000, 30000000, 2000000000000, 500000000000000, 2000000000000000, 5000000000, 600, "custom-1"),  # custom
                (400000, 145000000000000, 26000000, 45000000, 2000000000000, 230000000000000, 146000000000000, 5000000000, 600, "crv-eth-ethereum"),  # crv-eth; ethereum
                (4000, 1450000000000, 100000000, 150000000, 2000000000000, 230000000000000, 146000000000000, 5000000000, 600, "custom-2"),  # custom
            ]
            param_schema = [
                "A",
                "gamma",
                "mid_fee",
                "out_fee",
                "allowed_extra_profit",
                "fee_gamma",
                "adjustment_step",
                "admin_fee",
                "ma_half_time",
                "id"
            ]
            jobs = []
            params = [
                dict((k, v) for k, v in zip(param_schema, param))
                for param in param_values
            ]
            for idx, param in enumerate(params):
                param["name"] = f'curvev2_{param["id"]}'
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
            for concentration in [1]:
                param_values = [
                    (15, 1, 0, 5, 1,),  # no spread
                    (2.5, 3, 5, 5, 4),
                    (2.5, 6, 2.5, 5, 4),
                    (5, 1, 5, 5, 1),
                    (5, 3, 5, 5, 4),
                    (7.5, 2, 5, 5, 1),
                    (7.5, 1, 5, 5, 1),
                    (7.5, 1, 3, 5, 1),
                    (15, 1, 3, 5, 1),
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
                    param["name"] = f'{idx}{f"_conc{concentration}" if concentration != 1 else ""}'
                    param["concentration"] = concentration
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
            for concentration in [1, 10, 100]:
                deltas = [1, 5, 10, 30]
                jobs = []
                for delta in deltas:
                    lp_params = (typo, initial_inventories, initial_cash, (delta, concentration))
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

        elif "cfmmpowers" in typo:
            if typo == "cfmmpowers":
                color = "gray"
            elif typo == "cfmmpowers_noarb":
                color = "olive"
            else:
                raise ValueError("Unrecognized typo:", typo)
            deltas = [5, 30, 100]
            token0_weights = [10/100, 25/100, 75/100, 90/100]
            jobs = []
            for delta in deltas:
                for token0_weight in token0_weights:
                    assert 0 < token0_weight < 1
                    _weights = [token0_weight, 1 - token0_weight]
                    _initial_inventories = [2 * r * w for r, w in zip(initial_inventories, _weights)]
                    lp_params = (
                        typo, 
                        _initial_inventories, 
                        initial_cash, 
                        {
                            "delta": delta,
                            "weights": _weights
                        }
                    )
                    job = Process(target=monte_carlo,
                                args=(currencies_params, sizes, log_params, lp_params, simul_params, nb_MCs, seed, q))
                    job.start()
                    jobs.append(job)
                for job in jobs:
                    job.join()
            for delta in deltas:
                for token0_weight in token0_weights:
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
            gammas = [0., 10, 100., 5000., 10000]
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
            gammas = [0., 10, 100., 5000., 10000]
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
