import numpy as np
from sandbox import Logistic, MixedLogistics, LogisticExtended, MixedLogisticsExtended, Market,\
                   BaseOracle, PerfectOracle, LaggedOracle,\
                   CstDelta, CFMMPowers, CFMMSqrt, BestClosedForm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import warnings
warnings.filterwarnings("ignore")


currencies = ['BTC', 'ETH']
init_swap_price_01 = 3000. / 40000.
mu = 0.
scale = 1./252.
sigma = 0.2 * np.sqrt(scale)

sizes = np.array([1./40.])
nb_sizes = sizes.shape[0]

lambda_ = 45.
a = -1.8
b = 1300.

intensity_functions_01_object = [MixedLogistics(Logistic(lambda_, a, b), Logistic(lambda_/2., 0., 10.*b)) for _ in range(nb_sizes)]
intensity_functions_10_object = [MixedLogistics(Logistic(lambda_, a, b), Logistic(lambda_/2., 0., 10.*b)) for _ in range(nb_sizes)]

market = Market(currencies, init_swap_price_01, mu, sigma, sizes, intensity_functions_01_object, intensity_functions_10_object)

initial_inventories = 20. * np.array([1., 10. * 4./3.])
initial_cash = 0.

lps = []

sid = 24.*60.*60.


deltas = np.arange(1, 10, 1)
for delta in deltas:
    #lps.append(CstDelta('PO+%dbp' % delta, initial_inventories.copy(),
                                         #initial_cash, market, PerfectOracle(), delta * 1e-4))
    #lps.append(CstDelta('Lagged10+%dbp' % delta, initial_inventories.copy(),
                                         #initial_cash, market, LaggedOracle(10./sid), delta * 1e-4))
    lps.append(CFMMSqrt('CFMMSqrt+%dbp' % delta, initial_inventories.copy(), initial_cash, market,
                                        BaseOracle(), delta * 1e-4))
    lps.append(CFMMPowers('CFMMPowers0.3/0.7+%dbp' % delta, initial_inventories.copy(), initial_cash, market,
                                        BaseOracle(), np.array([0.3, 0.7]), delta * 1e-4))

ext = MixedLogisticsExtended(Logistic(lambda_, a, b), Logistic(lambda_/2., 0., 10.*b))
mq = ext.delta0 * 1e4
lps.append(CstDelta('PO+myopic', initial_inventories.copy(), initial_cash, market, PerfectOracle(), mq))

gammas = [0., 1., 5., 10., 50., 100., 500.]

for gamma in gammas:
    lp_BCF = BestClosedForm('PO+BCF%.0e' % gamma, initial_inventories.copy(), initial_cash,
                                             market, PerfectOracle(), gamma)
    lps.append(lp_BCF)

dt = 10./sid
T = 1.

for lp in lps:

    print('Liquidity provider: ', lp.name)

    np.random.seed(42)
    res = market.simulate(dt, T, lp, verbose=True)

    fig, axes = plt.subplots(5, 1)

    axes[0].plot(res.times, res.market_swap_prices, label=res.market.currencies[1]+res.market.currencies[0])
    axes[1].plot(res.times, res.cash, label='cash for ' + lp.name)
    axes[2].plot(res.times, res.inventories[:, 0], label=res.market.currencies[0] + ' inventory for ' + lp.name)
    axes[3].plot(res.times, res.inventories[:, 1], label=res.market.currencies[1] + ' inventory for ' + lp.name)
    axes[4].plot(res.times, res.pnl, label='PnL %s - PnL Hodl ' % lp.name)

    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    axes[3].legend()
    axes[4].legend()

    fig.show()
