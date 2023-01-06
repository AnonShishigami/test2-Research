import numpy as np
from scipy.stats import norm 

from sandbox.logistictools import LogisticTools
from sandbox.strategies.base.strategy import BaseLiquidityProvider


class SwaapV2(BaseLiquidityProvider):

    def __init__(self, name, initial_inventories, initial_cash, market, oracle, support_arb, gamma, 
        unpeg_tolerance=0.25/100,
        uncertainty_pricing_horizon=30/(24 * 3600),
        sigma_safety_margin=1.,
    ):

        super().__init__(name, initial_inventories, initial_cash, market, oracle, support_arb)
        
        self.sigma = self.market.sigma * sigma_safety_margin
        self.mu = self.market.mu

        self._target_inventories = self.initial_inventories.copy()
        self.collected_fees_01 = 0
        self.collected_fees_10 = 0
        self.unpeg_ratio = 1 + unpeg_tolerance
        self.delta_safety_margin = self.get_european_call_option_price(uncertainty_pricing_horizon, self.sigma)

        self.lts_01 = [LogisticTools(obj.lambda_, obj.a, obj.b) for obj in self.market.intensities_functions_01_object]
        self.lts_10 = [LogisticTools(obj.lambda_, obj.a, obj.b) for obj in self.market.intensities_functions_10_object]
        self.plus21 = np.sum(np.array([z * (self.lts_01[k].H_second(0.) + self.lts_10[k].H_second(0.))  for k,z in enumerate(self.market.sizes)]))
        self.minus22 = np.sum(np.array([z**2 * (self.lts_01[k].H_second(0.) - self.lts_10[k].H_second(0.)) for k, z in enumerate(self.market.sizes)]))
        self.minus11 = np.sum(np.array([z * (self.lts_01[k].H_prime(0.) - self.lts_10[k].H_prime(0.)) for k, z in enumerate(self.market.sizes)]))
        self.update_gamma(gamma)

    def update_gamma(self, gamma):
        self.gamma = gamma
        self.a = (self.sigma**2 + np.sqrt(self.sigma**4 + 4. * self.gamma * self.sigma**2 * self.plus21)) / (4. * self.plus21)
        self.b = - (0.5 * self.mu / self.a + self.minus11 - self.minus22 * self.a) / self.plus21

    def pricing_function_01(self, nb_coins_1, swap_price_01):

        if self.inventories[1] < nb_coins_1:
            return np.inf, 0., 0.

        hodl_spread = (self.inventories[1] - self._target_inventories[1]) * swap_price_01
        size = nb_coins_1 * swap_price_01
        z_index = np.argmin(np.abs(self.market.sizes-size))
        p = -2. * self.a * hodl_spread - self.b + size * self.a
        delta = self.lts_01[z_index].delta(p)
        delta = max(0, delta)
        delta += self.delta_safety_margin
        self.collected_fees_01 = delta * nb_coins_1 * swap_price_01
        return swap_price_01 * (1. + delta), 0, 0

    def pricing_function_10(self, nb_coins_0, swap_price_10):

        if self.inventories[0] < nb_coins_0:
            return np.inf, 0., 0.

        hodl_spread = (self.inventories[1] - self._target_inventories[1]) / swap_price_10
        size = nb_coins_0
        z_index = np.argmin(np.abs(self.market.sizes-size))
        p = 2. * self.a * hodl_spread + self.b + size * self.a
        delta = self.lts_10[z_index].delta(p)
        delta = max(0, delta)
        delta += self.delta_safety_margin
        self.collected_fees_10 = delta * nb_coins_0 * swap_price_10
        return swap_price_10 * (1. + delta), 0, 0

    def update_01(self, trade_01):
        success = super().update_01(trade_01)
        if not success:
            return False
        if trade_01 == 1:
            self._target_inventories[0] += self.collected_fees_01
            self.update_gamma(self.gamma * self._gamma_factor())
        return True

    def update_10(self, trade_10):
        success = super().update_10(trade_10)
        if not success:
            return False
        if trade_10 == 1:
            self._target_inventories[1] += self.collected_fees_10
            self.update_gamma(self.gamma * self._gamma_factor())
        return True

    @staticmethod
    def d1(S, K, r, sigma, T):
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def d2(S, K, r, sigma, T):
        return (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    def _european_call_option_pricing(self, S, K, r, sigma, T):
        return S * norm.cdf(self.d1(S, K, r, sigma, T), 0.0, 1.0) - K * np.exp(-r * T) * norm.cdf(self.d2(S, K, r, sigma, T), 0.0, 1.0)

    def get_european_call_option_price(self, T, sigma):
        if T == 0:
            return 0
        S = 1
        K = S
        r = 5 / 100
        return self._european_call_option_pricing(S, K, r, sigma, T) / S
    
    def _gamma_factor(self):
        price = self.oracle.get()
        initial = self._tvl(price, self.initial_inventories)
        current = self._tvl(price, self._target_inventories)
        return initial / current
    
    @staticmethod
    def _tvl(price, reserves):
        return reserves[0] + reserves[1] * price

    def get_state(self):
        return super().get_state() | {
            "target_inventories": self._target_inventories,
        }

    def restore_state(self, state):
        super().restore_state(state)
        self._target_inventories = state["target_inventories"]
