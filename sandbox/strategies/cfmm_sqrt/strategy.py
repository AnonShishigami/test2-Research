import numpy as np

from sandbox.strategies.cfmm_powers.strategy import CFMMPowers


class CFMMSqrt(CFMMPowers):

    def __init__(self, name, initial_inventories, initial_cash, market, oracle, support_arb, delta):
        super().__init__(name, initial_inventories, initial_cash, market, oracle, support_arb, np.array([0.5, 0.5]), delta)

    def _arb_01(self, time, swap_price_01, relative_cost, fixed_cost, a, *args, **kwargs):
        amount = self.concentrated_inventories[1] - np.sqrt(
            self.concentrated_inventories[0] * self.concentrated_inventories[1] / (swap_price_01 / (1 + relative_cost) * (1 - self.delta))
        )
        if amount > 0:
            self.proposed_swap_prices_01(time, amount)
            self.update_01(1)
        return amount

    def _arb_10(self, time, swap_price_10, relative_cost, fixed_cost, a, *args, **kwargs):
        amount = self.concentrated_inventories[0] - np.sqrt(
            self.concentrated_inventories[0] * self.concentrated_inventories[1] / (swap_price_10 / (1 + relative_cost) * (1 - self.delta))
        )
        if amount > 0:
            self.proposed_swap_prices_10(time, amount)
            self.update_10(1)
        return amount