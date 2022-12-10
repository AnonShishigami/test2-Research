import numpy as np

from sandbox.logistictools import LogisticTools
from sandbox.strategies.base.strategy import BaseLiquidityProvider


class BestClosedForm(BaseLiquidityProvider):

    def __init__(self, name, initial_inventories, initial_cash, market, oracle, support_arb, gamma):

        super().__init__(name, initial_inventories, initial_cash, market, oracle, support_arb)
        self.gamma = gamma

        self.lts_01 = [LogisticTools(obj.lambda_, obj.a, obj.b) for obj in self.market.intensities_functions_01_object]
        self.lts_10 = [LogisticTools(obj.lambda_, obj.a, obj.b) for obj in self.market.intensities_functions_10_object]

        plus21 = np.sum(np.array([z * (self.lts_01[k].H_second(0.) + self.lts_10[k].H_second(0.))  for k,z in enumerate(self.market.sizes)]))
        minus22 = np.sum(np.array([z**2 * (self.lts_01[k].H_second(0.) - self.lts_10[k].H_second(0.)) for k, z in enumerate(self.market.sizes)]))
        minus11 = np.sum(np.array([z * (self.lts_01[k].H_prime(0.) - self.lts_10[k].H_prime(0.)) for k, z in enumerate(self.market.sizes)]))

        self.a = (self.market.sigma**2 + np.sqrt(self.market.sigma**4 + 4. * self.gamma * self.market.sigma**2 * plus21)) / (4. * plus21)
        self.b = - (0.5 * self.market.mu / self.a + minus11 - minus22 * self.a) / plus21

    def pricing_function_01(self, nb_coins_1, swap_price_01):

        if self.inventories[1] < nb_coins_1:
            return np.inf, 0.

        hodl_spread = (self.inventories[1] - self.initial_inventories[1]) * swap_price_01
        size = nb_coins_1 * swap_price_01
        z_index = np.argmin(np.abs(self.market.sizes-size))
        p = -2. * self.a * hodl_spread - self.b + size * self.a
        delta = self.lts_01[z_index].delta(p)
        return swap_price_01 * (1. + delta), delta * nb_coins_1 * swap_price_01

    def pricing_function_10(self, nb_coins_0, swap_price_10):

        if self.inventories[0] < nb_coins_0:
            return np.inf, 0.

        hodl_spread = (self.inventories[1] - self.initial_inventories[1]) / swap_price_10
        size = nb_coins_0
        z_index = np.argmin(np.abs(self.market.sizes-size))
        p = 2. * self.a * hodl_spread + self.b + size * self.a
        delta = self.lts_10[z_index].delta(p)
        return swap_price_10 * (1. + delta), delta * nb_coins_0

