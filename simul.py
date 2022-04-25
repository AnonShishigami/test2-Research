import numpy as np
from multiprocessing import Process
import pickle
from ammlib import Logistic, Market, LiquidityProviderCstDelta, LiquidityProviderAMMSqrt, LogisticTools
from ammlib.lpopt import LiquidityProviderHJB, NumericalParams
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import warnings
warnings.filterwarnings("ignore")


currencies = ['BTC', 'ETH']
init_prices = np.array([40000., 3000.])
mu = np.zeros(2)
scale = 1./252.
volatilities = np.array([0.4, 0.4])
rho = 0.5
Sigma = scale * np.array([[volatilities[0]**2, rho * volatilities[0] * volatilities[1]],
                  [rho * volatilities[0] * volatilities[1], volatilities[1]**2]])
sizes = np.array([1000., 2000.])
nb_sizes = sizes.shape[0]

lambda_ = 100.
a = 0.
b = 2e3

intensity_functions_01_object = [Logistic(lambda_, a, b) for _ in range(nb_sizes)]
intensity_functions_10_object = [Logistic(lambda_, a, b) for _ in range(nb_sizes)]

market = Market(currencies, init_prices, mu, Sigma, sizes, intensity_functions_01_object, intensity_functions_10_object)

initial_inventories = 20. * np.array([1., 10. * 4./3.])
initial_cash = 0.

lps = []


deltas = np.arange(1, 25, 1)
for delta in deltas:
    lps.append(LiquidityProviderCstDelta('cst%d' % delta, initial_inventories.copy(),
                                         initial_cash, market, delta * 1e-4))

for delta in deltas:
    lps.append(LiquidityProviderAMMSqrt('cfmm%d' % delta, initial_inventories.copy(), initial_cash, market,
                                        delta * 1e-4))

lt = LogisticTools(lambda_, a, b)
mq = lt.myopic_quote

lps.append(LiquidityProviderCstDelta('myopic', initial_inventories.copy(), initial_cash, market, mq))



plt.rcParams["figure.figsize"] = [16, 9]

dt = 10./(24.*60.*60.)
T = 1.

for lp in lps:

    print('Liquidity provider: ', lp.name)

    np.random.seed(42)
    res = market.simulate(dt, T, lp, verbose=True)

    fig, axes = plt.subplots(5, 1)

    axes[0].plot(res.times, res.prices[:, 0], label=res.market.currencies[0] + ' price')
    axes[1].plot(res.times, res.prices[:, 1], label=res.market.currencies[1] + ' price')
    axes[2].plot(res.times, res.inventories[:, 0], label=res.market.currencies[0] + ' inventory for ' + lp.name)
    axes[3].plot(res.times, res.inventories[:, 1], label=res.market.currencies[1] + ' inventory for ' + lp.name)
    axes[4].plot(res.times, res.pnl, label='PnL %s - PnL Hodl ' % lp.name)

    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    axes[3].legend()
    axes[4].legend()

    fig.show()