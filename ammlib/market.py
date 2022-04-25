import numpy as np


class SimulationResults:

    def __init__(self, times, prices, inventories, cash, pnl, market):

        self.times = times
        self.prices = prices
        self.inventories = inventories
        self.cash = cash
        self.pnl = pnl
        self.market = market


class Market:

    def __init__(self, currencies, init_prices, mu, Sigma, sizes,
                 intensity_functions_01_object, intensity_functions_10_object):

        self.currencies = currencies
        self.init_prices = init_prices.copy()
        self.swap_price01 = init_prices[1] / init_prices[0]  # in currency 0
        self.swap_price10 = init_prices[0] / init_prices[1]  # in currency 1
        self.mu = mu
        self.Sigma = Sigma
        self.sizes = sizes  # in reference currency
        self.nb_sizes = sizes.shape[0]
        self.intensities_functions_01_object = intensity_functions_01_object
        self.intensities_functions_10_object = intensity_functions_10_object
        self.intensities_functions_01 = [obj.Lambda for obj in intensity_functions_01_object]
        self.intensities_functions_10 = [obj.Lambda for obj in intensity_functions_10_object]

    def simulate(self, dt, T, lp, verbose=False):

        nb_t = int(T / dt) + 1
        times = np.linspace(0., T, nb_t)

        inventories = np.zeros([nb_t, 2])
        cash = np.zeros(nb_t)
        mtm_value = np.zeros(nb_t)
        mtm_value_hodl = np.zeros(nb_t)

        inventories[0] = lp.inventories
        cash[0] = lp.cash
        mtm_value[0] = lp.mtm_value(self.init_prices)
        mtm_value_hodl[0] = lp.initial_cash + np.inner(self.init_prices, lp.initial_inventories)

        prices = np.zeros([nb_t, 2])
        prices[1:, :] = np.sqrt(dt) * np.cumsum(np.random.multivariate_normal(np.array([0., 0.]), self.Sigma, nb_t-1),
                                                axis=0)
        prices[:, 0] = self.init_prices[0] * np.exp(
            (self.mu[0] - self.Sigma[0, 0]/2.) * times + prices[:, 0])
        prices[:, 1] = self.init_prices[1] * np.exp(
            (self.mu[1] - self.Sigma[1, 1] / 2.) * times + prices[:, 1])
        swap_prices_01 = prices[:, 1] / prices[:, 0]
        swap_prices_10 = prices[:, 0] / prices[:, 1]

        current_prices = prices[0]
        current_swap_price_01 = swap_prices_01[0]
        current_swap_price_10 = swap_prices_10[0]

        if verbose:
            print("C'est parti pour %d périodes\n"%nb_t)

        for t in range(nb_t-1):

            if verbose and t % 100 == 0:
                print(t)

            permutation = np.random.permutation(2*self.nb_sizes)

            for index in permutation:

                side = index % 2
                size_index = index // 2
                size = self.sizes[size_index]

                if side == 0:

                    proposed_swap_price_01 = lp.proposed_swap_prices_01(size / current_prices[1], current_prices)
                    proposed_swap_delta_01 = (proposed_swap_price_01 - current_swap_price_01) / current_swap_price_01
                    proba_01 = 1. - np.exp(- self.intensities_functions_01[size_index](proposed_swap_delta_01) * dt)
                    trade_01 = np.random.binomial(1, proba_01)
                    lp.update_01(trade_01)

                else:

                    proposed_swap_price_10 = lp.proposed_swap_prices_10(size / current_prices[0], current_prices)
                    proposed_swap_delta_10 = (proposed_swap_price_10 - current_swap_price_10) / current_swap_price_10
                    proba_10 = 1. - np.exp(- self.intensities_functions_10[size_index](proposed_swap_delta_10) * dt)
                    trade_10 = np.random.binomial(1, proba_10)
                    lp.update_10(trade_10)

            current_prices = prices[t + 1]
            current_swap_price_01 = swap_prices_01[t + 1]
            current_swap_price_10 = swap_prices_10[t + 1]

            inventories[t + 1, :] = lp.inventories
            cash[t + 1] = lp.cash
            mtm_value[t + 1] = lp.mtm_value(current_prices)
            mtm_value_hodl[t + 1] = lp.initial_cash + np.inner(current_prices, lp.initial_inventories)

        return SimulationResults(times, prices, inventories, cash, mtm_value - mtm_value_hodl, self)
