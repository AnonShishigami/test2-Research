import numpy as np

from sandbox.strategies.base.strategy import BaseLiquidityProvider


class CFMMPowers(BaseLiquidityProvider):

    def __init__(self, name, initial_inventories, initial_cash, market, oracle, support_arb, weights, delta, concentration=1):
        super().__init__(name, initial_inventories, initial_cash, market, oracle, support_arb)
        self.delta = delta
        self.w0 = weights[0]
        self.w1 = weights[1]
        self.concentration = concentration
        self.concentrated_inventories = [i * np.sqrt(self.concentration) for i in self.inventories]

    def pricing_function_01(self, nb_coins_1, swap_price_01):
        if self.inventories[1] <= nb_coins_1:
            return np.inf, 0.
        else:
            return self.get_cfmm_powers_price(
                self.concentrated_inventories[0], self.w0, self.concentrated_inventories[1], self.w1, nb_coins_1, self.delta, price=swap_price_01
            ), 0.

    def pricing_function_10(self, nb_coins_0, swap_price_10):
        if self.inventories[0] <= nb_coins_0:
            return np.inf, 0.
        else:
            return self.get_cfmm_powers_price(
                self.concentrated_inventories[1], self.w1, self.concentrated_inventories[0], self.w0, nb_coins_0, self.delta, price=swap_price_10
            ), 0.

    @staticmethod
    def get_cfmm_powers_amount_out(r_in, w_in, r_out, w_out, amount_out, delta, liquidity):
        return (((liquidity / (r_out - amount_out) ** w_out) ** (1 / w_in) - r_in)) / (1. - delta)

    def get_cfmm_powers_price(self, r_in, w_in, r_out, w_out, amount_out, delta, *args, **kwargs):
        liquidity = self.compute_liquidity(r_in, w_in, r_out, w_out, *args, **kwargs)
        return self.get_cfmm_powers_amount_out(r_in, w_in, r_out, w_out, amount_out, delta, liquidity) / amount_out

    @staticmethod
    def compute_liquidity(r_in, w_in, r_out, w_out, *args, **kwargs):
        return r_in ** w_in  * r_out ** w_out

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
        self.concentrated_inventories = state["concentrated_inventories"]

    def get_state(self):
        return super().get_state() | {
            "concentrated_inventories": [float(v) for v in self.concentrated_inventories],
        }