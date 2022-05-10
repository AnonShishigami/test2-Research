import numpy as np
from .logistictools import LogisticTools
import matplotlib.pyplot as plt

class BaseLiquidityProvider:

    def __init__(self, name, initial_inventories, initial_cash, market, oracle):

        self.name = name
        self.initial_inventories = initial_inventories.copy()
        self.initial_cash = initial_cash
        self.inventories = initial_inventories
        self.cash = initial_cash
        self.oracle = oracle
        self.market = market
        self.last_requested_nb_coins_0 = None
        self.last_requested_nb_coins_1 = None
        self.last_answer_01 = None
        self.last_answer_10 = None
        self.last_cashed_01 = None
        self.last_cashed_10 = None


    def reset(self):

        self.inventories = self.initial_inventories.copy()
        self.cash = self.initial_cash
        self.oracle.reset()

    def pricing_function_01(self, nb_coins_1, swap_price_01):
        return np.inf, 0.

    def pricing_function_10(self, nb_coins_0, swap_price_10):
        return np.inf, 0.

    def proposed_swap_prices_01(self, time, nb_coins_1):

        self.last_requested_nb_coins_1 = nb_coins_1
        swap_price_01 = self.oracle.get(time)
        answer, cashed = self.pricing_function_01(nb_coins_1, swap_price_01)
        self.last_answer_01 = answer
        self.last_cashed_01 = cashed

        return answer

    def proposed_swap_prices_10(self, time, nb_coins_0):

        self.last_requested_nb_coins_0 = nb_coins_0
        swap_price_10 = 1. / self.oracle.get(time)
        answer, cashed = self.pricing_function_10(nb_coins_0, swap_price_10)
        self.last_answer_10 = answer
        self.last_cashed_10 = cashed

        return answer

    def update_01(self, trade_01):

        if trade_01 == 1:
            self.inventories[0] += self.last_requested_nb_coins_1 * self.last_answer_01 - self.last_cashed_01
            self.inventories[1] -= self.last_requested_nb_coins_1
            self.cash += self.last_cashed_01

    def update_10(self, trade_10):

        if trade_10 == 1:
            self.inventories[1] += self.last_requested_nb_coins_0 * self.last_answer_10
            self.inventories[0] -= self.last_requested_nb_coins_0 - self.last_cashed_10
            self.cash += self.last_cashed_10

    def mtm_value(self, swap_price_01):
        return self.cash + self.inventories[0] + swap_price_01 * self.inventories[1]


class LiquidityProviderCstDelta(BaseLiquidityProvider):

    def __init__(self, name, initial_inventories, initial_cash, market, oracle, delta):
        super().__init__(name, initial_inventories, initial_cash, market, oracle)
        self.delta = delta

    def pricing_function_01(self, nb_coins_1, swap_price_01):
        if self.inventories[1] < nb_coins_1:
            return np.inf, 0.
        else:
            return swap_price_01 * (1. + self.delta), self.delta * nb_coins_1 * swap_price_01

    def pricing_function_10(self, nb_coins_0, swap_price_10):
        if self.inventories[0] < nb_coins_0:
            return np.inf, 0.
        else:
            return swap_price_10 * (1. + self.delta), self.delta * nb_coins_0


class LiquidityProviderAMMSqrt(BaseLiquidityProvider):

    def __init__(self, name, initial_inventories, initial_cash, market, oracle, delta):
        super().__init__(name, initial_inventories, initial_cash, market, oracle)
        self.delta = delta

    def pricing_function_01(self, nb_coins_1, swap_price_01):
        if self.inventories[1] <= nb_coins_1:
            return np.inf, 0.
        else:
            return self.inventories[0] / (self.inventories[1] - nb_coins_1) / (1. - self.delta), 0.

    def pricing_function_10(self, nb_coins_0, swap_price_10):
        if self.inventories[0] <= nb_coins_0:
            return np.inf, 0.
        else:
            return self.inventories[1] / (self.inventories[0] - nb_coins_0) / (1. - self.delta), 0.


class LiquidityProviderBestClosedForm(BaseLiquidityProvider):

    def __init__(self, name, initial_inventories, initial_cash, market, oracle, gamma):

        super().__init__(name, initial_inventories, initial_cash, market, oracle)
        self.gamma = gamma

        self.lts_01 = [LogisticTools(obj.lambda_, obj.a, obj.b) for obj in self.market.intensities_functions_01_object]
        self.lts_10 = [LogisticTools(obj.lambda_, obj.a, obj.b) for obj in self.market.intensities_functions_10_object]

        plus21 = np.sum(np.array([z * (self.lts_01[k].H_second(0.) + self.lts_10[k].H_second(0.))  for k,z in enumerate(self.market.sizes)]))
        minus22 = np.sum(np.array([z**2 * (self.lts_01[k].H_second(0.) - self.lts_10[k].H_second(0.)) for k, z in enumerate(self.market.sizes)]))
        minus11 = np.sum(np.array([z * (self.lts_01[k].H_prime(0.) - self.lts_10[k].H_prime(0.)) for k, z in enumerate(self.market.sizes)]))

        self.a = (self.market.sigma**2 + np.sqrt(self.market.sigma**4 + 4. * self.gamma * self.market.sigma**2 * plus21)) / (4. * plus21)
        self.b = - (minus11 - minus22 * self.a) / plus21

    def pricing_function_01(self, nb_coins_1, swap_price_01):

        if self.inventories[1] < nb_coins_1:
            return np.inf, 0.

        else:
            hodl_spread = (self.inventories[1] - self.initial_inventories[1]) * swap_price_01
            size = nb_coins_1 * swap_price_01
            z_index =  np.argmin(np.abs(self.market.sizes-size))
            p = -2. * self.a * hodl_spread - self.b + size * self.a
            delta = self.lts_01[z_index].delta(p)
            return swap_price_01 * (1. + delta), delta * nb_coins_1 * swap_price_01

    def pricing_function_10(self, nb_coins_0, swap_price_10):

        if self.inventories[0] < nb_coins_0:
            return np.inf, 0.

        else:
            hodl_spread = (self.inventories[1] - self.initial_inventories[1]) / swap_price_10
            size = nb_coins_0
            z_index = np.argmin(np.abs(self.market.sizes-size))
            p = 2. * self.a * hodl_spread + self.b + size * self.a
            delta = self.lts_10[z_index].delta(p)

        return swap_price_10 * (1. + delta), delta * nb_coins_0