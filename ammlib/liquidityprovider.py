import numpy as np


class BaseLiquidityProvider:

    def __init__(self, name, initial_inventories, initial_cash, market):

        self.name = name
        self.initial_inventories = initial_inventories.copy()
        self.initial_cash = initial_cash
        self.inventories = initial_inventories
        self.cash = initial_cash
        self.market = market
        self.last_requested_nb_coins_0 = None
        self.last_requested_nb_coins_1 = None
        self.last_answer_01 = None
        self.last_answer_10 = None

    def reset(self):

        self.inventories = self.initial_inventories.copy()
        self.cash = self.initial_cash

    def pricing_function_01(self, nb_coins_1, prices):
        return np.inf

    def pricing_function_10(self, nb_coins_0, prices):
        return np.inf

    def proposed_swap_prices_01(self, nb_coins_1, prices):

        self.last_requested_nb_coins_1 = nb_coins_1
        answer = self.pricing_function_01(nb_coins_1, prices)
        self.last_answer_01 = answer

        return answer

    def proposed_swap_prices_10(self, nb_coins_0, prices):

        self.last_requested_nb_coins_0 = nb_coins_0
        answer = self.pricing_function_10(nb_coins_0, prices)
        self.last_answer_10 = answer

        return answer

    def update_01(self, trade_01):

        if trade_01 == 1:
            self.inventories[0] += self.last_requested_nb_coins_1 * self.last_answer_01
            self.inventories[1] -= self.last_requested_nb_coins_1

    def update_10(self, trade_10):

        if trade_10 == 1:
            self.inventories[1] += self.last_requested_nb_coins_0 * self.last_answer_10
            self.inventories[0] -= self.last_requested_nb_coins_0

    def mtm_value(self, prices):
        return self.cash + np.inner(prices, self.inventories)


class LiquidityProviderCstDelta(BaseLiquidityProvider):

    def __init__(self, name, initial_inventories, initial_cash, market, delta):
        super().__init__(name, initial_inventories, initial_cash, market)
        self.delta = delta

    def pricing_function_01(self, nb_coins_1, prices):
        swap_price_01 = prices[1] / prices[0]
        if self.inventories[1] < nb_coins_1:
            return np.inf
        else:
            return swap_price_01 * (1. + self.delta)

    def pricing_function_10(self, nb_coins_0, prices):
        swap_price_10 = prices[0] / prices[1]
        if self.inventories[0] < nb_coins_0:
            return np.inf
        else:
            return swap_price_10 * (1. + self.delta)


class LiquidityProviderAMMSqrt(BaseLiquidityProvider):

    def __init__(self, name, initial_inventories, initial_cash, market, delta):
        super().__init__(name, initial_inventories, initial_cash, market)
        self.delta = delta

    def pricing_function_01(self, nb_coins_1, prices):
        if self.inventories[1] <= nb_coins_1:
            return np.inf
        else:
            return self.inventories[0] / (self.inventories[1] - nb_coins_1) / (1. - self.delta)

    def pricing_function_10(self, nb_coins_0, prices):
        if self.inventories[0] <= nb_coins_0:
            return np.inf
        else:
            return self.inventories[1] / (self.inventories[0] - nb_coins_0) / (1. - self.delta)
