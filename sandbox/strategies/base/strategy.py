import copy

import numpy as np


class BaseLiquidityProvider:

    def __init__(self, name, initial_inventories, initial_cash, market, oracle, support_arb):

        self.name = name
        self.initial_inventories = initial_inventories.copy()
        self.initial_cash = initial_cash
        self.inventories = initial_inventories
        self.cash = initial_cash
        self.oracle = oracle
        self.market = market
        self.support_arb = support_arb
        self.last_requested_nb_coins_0 = None
        self.last_requested_nb_coins_1 = None
        self.last_answer_01 = None
        self.last_answer_10 = None
        self.last_cashed_01 = None
        self.last_cashed_10 = None

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
        return True

    def update_10(self, trade_10):

        if trade_10 == 1:
            self.inventories[1] += self.last_requested_nb_coins_0 * self.last_answer_10
            self.inventories[0] -= self.last_requested_nb_coins_0 - self.last_cashed_10
            self.cash += self.last_cashed_10
        return True

    def mtm_value(self, swap_price_01):
        return self.cash + self.inventories[0] + swap_price_01 * self.inventories[1]

    def arb_01(self, time, swap_price_01, relative_cost, fixed_cost, step_ratio=10000, *args, **kwargs):
        if not self.support_arb:
            return 0
        return self._arb_01(time, swap_price_01, relative_cost, fixed_cost, step_ratio, *args, **kwargs)

    def arb_10(self, time, swap_price_10, relative_cost, fixed_cost, step_ratio=10000, *args, **kwargs):
        if not self.support_arb:
            return 0
        return self._arb_10(time, swap_price_10, relative_cost, fixed_cost, step_ratio, *args, **kwargs)

    def _arb_01(self, time, swap_price_01, relative_cost, fixed_cost, step_ratio, *args, **kwargs):

        state = self.get_state()

        s = self.inventories[1] / step_ratio
        amount = 0
        ko = 0
        ok = 0
        while ko < 4 and ok < 100:
            proposed_swap_price_01 = self.proposed_swap_prices_01(time, s)
            if proposed_swap_price_01 * (1 + relative_cost) > swap_price_01:
                s /= 2
                ko += 1
            else:
                success = self.update_01(1)
                if not success:
                    break
                amount += s
                ok += 1

        if amount:
            self.restore_state(state)
            proposed_swap_price_01 = self.proposed_swap_prices_01(time, amount)
            if proposed_swap_price_01 * (1 + relative_cost) <= swap_price_01:
                self.update_01(1)

        return amount

    def _arb_10(self, time, swap_price_10, relative_cost, fixed_cost, step_ratio, *args, **kwargs):

        state = self.get_state()

        s = self.inventories[0] / step_ratio
        amount = 0
        ko = 0
        ok = 0
        while ko < 4 and ok < 100:
            proposed_swap_price_10 = self.proposed_swap_prices_10(time, s)
            if proposed_swap_price_10 * (1 + relative_cost) > swap_price_10:
                s /= 2
                ko += 1
            else:
                success = self.update_10(1)
                if not success:
                    break
                amount += s
                ok += 1

        if amount:
            self.restore_state(state)
            proposed_swap_price_10 = self.proposed_swap_prices_10(time, amount)
            if proposed_swap_price_10 * (1 + relative_cost) <= swap_price_10:
                self.update_10(1)

        return amount

    def restore_state(self, state):
        self.inventories = state["inventories"]
        self.cash = state["cash"]

    def get_state(self):
        return {
            "inventories": copy.deepcopy([float(v) for v in self.inventories]),
            "cash": float(self.cash),
        }