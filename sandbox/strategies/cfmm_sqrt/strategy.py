import numpy as np

from sandbox.strategies.cfmm_powers.strategy import CFMMPowers


class CFMMSqrt(CFMMPowers):

    def __init__(self, name, initial_inventories, initial_cash, market, oracle, support_arb, delta, concentration=1):
        super().__init__(name, initial_inventories, initial_cash, market, oracle, support_arb, np.array([0.5, 0.5]), delta)

        self.concentration = concentration
        self.concentrated_inventories = [i * np.sqrt(self.concentration) for i in self.inventories]

    def _get_pricing_inventories(self):
        return self.concentrated_inventories

    def _arb_01(self, swap_price_01, relative_cost, fixed_cost, *args, **kwargs):
        pricing_inventories = self._get_pricing_inventories()
        amount = pricing_inventories[1] - np.sqrt(
            pricing_inventories[0] * pricing_inventories[1] / (swap_price_01 / (1 + relative_cost) * (1 - self.delta))
        )
        if amount > 0:
            self.proposed_swap_prices_01(amount)
            self.update_01(1)
        return amount

    def _arb_10(self, swap_price_10, relative_cost, fixed_cost, *args, **kwargs):
        pricing_inventories = self._get_pricing_inventories()
        amount = pricing_inventories[0] - np.sqrt(
            pricing_inventories[0] * pricing_inventories[1] / (swap_price_10 / (1 + relative_cost) * (1 - self.delta))
        )
        if amount > 0:
            self.proposed_swap_prices_10(amount)
            self.update_10(1)
        return amount

    def update_01(self, trade_01):
        before = [v for v in self.inventories]
        success = super().update_01(trade_01)
        if not success:
            return False
        if trade_01 == 1:
            for i in range(len(before)):
                self.concentrated_inventories[i] += self.inventories[i] - before[i]
            self.cash += self.last_cashed_01
        return True 

    def update_10(self, trade_10):
        before = [v for v in self.inventories]
        success = super().update_10(trade_10)
        if not success:
            return False
        if trade_10 == 1:
            for i in range(len(before)):
                self.concentrated_inventories[i] += self.inventories[i] - before[i]
            self.cash += self.last_cashed_10
        return True
    
    def restore_state(self, state):
        super().restore_state(state)
        self.concentrated_inventories = np.array(state["concentrated_inventories"])

    def get_state(self):
        s = super().get_state()
        s["concentrated_inventories"] = [float(v) for v in self.concentrated_inventories]
        return s