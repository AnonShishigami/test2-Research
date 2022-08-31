import numpy as np

from sandbox.strategies.base.strategy import BaseLiquidityProvider


class CstDelta(BaseLiquidityProvider):

    def __init__(self, name, initial_inventories, initial_cash, market, oracle, support_arb, delta):
        super().__init__(name, initial_inventories, initial_cash, market, oracle, support_arb)
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

    def _arb_01(self, time, swap_price_01, relative_cost, fixed_cost, *args, **kwargs):
        if self.oracle.get(time) * (1 + relative_cost) > swap_price_01:
            return 0
        amount = self.inventories[1]
        self.proposed_swap_prices_01(time, amount)
        self.update_01(1)
        return amount

    def _arb_10(self, time, swap_price_10, relative_cost, fixed_cost, *args, **kwargs):
        if 1. / self.oracle.get(time) * (1 + relative_cost) > swap_price_10:
            return 0
        amount = self.inventories[0]
        self.proposed_swap_prices_10(time, amount)
        self.update_10(1)
        return amount
