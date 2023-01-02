import numpy as np

from sandbox.strategies.base.strategy import BaseLiquidityProvider


class Clipper(BaseLiquidityProvider):

    def __init__(self, name, initial_inventories, initial_cash, market, oracle, support_arb, k, delta):

        super().__init__(name, initial_inventories, initial_cash, market, oracle, support_arb)
        self.k = k
        self.delta = delta

    def pricing_function_01(self, nb_coins_1, swap_price_01):
        if self.inventories[1] <= nb_coins_1:
            return np.inf, 0., 0.
        else:
            invariant = self.inventories[0] ** (1 - self.k) + (swap_price_01 * self.inventories[1]) ** (1 - self.k)
            input_amount = ((invariant - (swap_price_01 * (self.inventories[1] - nb_coins_1)) ** (1 - self.k)) ** (1 / (1 - self.k)) - self.inventories[0]) / (1. - self.delta)
            return input_amount / nb_coins_1, 0, 0

    def pricing_function_10(self, nb_coins_0, swap_price_10):
        if self.inventories[0] <= nb_coins_0:
            return np.inf, 0., 0.
        else:
            invariant = self.inventories[0] ** (1 - self.k) + (1 / swap_price_10 * self.inventories[1]) ** (1 - self.k)
            input_amount = ((invariant - (self.inventories[0] - nb_coins_0) ** (1 - self.k)) ** (1 / (1 - self.k)) * swap_price_10 - self.inventories[1]) / (1. - self.delta)
            return input_amount / nb_coins_0, 0, 0
