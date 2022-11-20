import numpy as np

from sandbox.strategies.base.strategy import BaseLiquidityProvider


class CFMMPowers(BaseLiquidityProvider):

    def __init__(self, name, initial_inventories, initial_cash, market, oracle, support_arb, weights, delta):
        super().__init__(name, initial_inventories, initial_cash, market, oracle, support_arb)
        self.delta = delta
        self.w0 = weights[0]
        self.w1 = weights[1]

    def pricing_function_01(self, nb_coins_1, swap_price_01):
        if self.inventories[1] <= nb_coins_1:
            return np.inf, 0.
        else:
            return self.get_cfmm_powers_price(
                self._get_pricing_inventories()[0], self.w0, self._get_pricing_inventories()[1], self.w1, nb_coins_1, self.delta, price=swap_price_01
            ), 0.

    def pricing_function_10(self, nb_coins_0, swap_price_10):
        if self.inventories[0] <= nb_coins_0:
            return np.inf, 0.
        else:
            return self.get_cfmm_powers_price(
                self._get_pricing_inventories()[1], self.w1, self._get_pricing_inventories()[0], self.w0, nb_coins_0, self.delta, price=swap_price_10
            ), 0.

    @staticmethod
    def get_cfmm_powers_amount_out(r_in, w_in, r_out, w_out, amount_out, delta, liquidity):
        return (((liquidity / (r_out - amount_out) ** w_out) ** (1 / w_in) - r_in)) / (1. - delta)

    def get_cfmm_powers_price(self, r_in, w_in, r_out, w_out, amount_out, delta, *args, **kwargs):
        liquidity = self.compute_liquidity(r_in, w_in, r_out, w_out, *args, **kwargs)
        return self.get_cfmm_powers_amount_out(r_in, w_in, r_out, w_out, amount_out, delta, liquidity) / amount_out

    @staticmethod
    def compute_liquidity(r_in, w_in, r_out, w_out, *args, **kwargs):
        return r_in ** w_in * r_out ** w_out

    def _get_pricing_inventories(self):
        return self.inventories