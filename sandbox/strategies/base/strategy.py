import copy

import numpy as np


class BaseLiquidityProvider:

    def __init__(self, name, initial_inventories, initial_cash, market, oracle, support_arb):

        self.name = name
        self.initial_inventories = initial_inventories.copy()
        self.initial_cash = initial_cash
        self.inventories = initial_inventories.copy()
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
        self._pause = False
        self.unpeg_ratio = None

    def pricing_function_01(self, nb_coins_1, swap_price_01):
        return np.inf, 0., 0.

    def pricing_function_10(self, nb_coins_0, swap_price_10):
        return np.inf, 0., 0.

    def proposed_swap_prices_01(self, nb_coins_1):

        self.last_requested_nb_coins_1 = nb_coins_1
        swap_price_01 = self.oracle.get()
        answer, cashed, giveback = self.pricing_function_01(nb_coins_1, swap_price_01)
        self.last_answer_01 = answer
        self.last_cashed_01 = cashed
        self.giveback_01 = giveback

        return answer

    def proposed_swap_prices_10(self, nb_coins_0):

        self.last_requested_nb_coins_0 = nb_coins_0
        swap_price_10 = 1. / self.oracle.get()
        answer, cashed, giveback = self.pricing_function_10(nb_coins_0, swap_price_10)
        self.last_answer_10 = answer
        self.last_cashed_10 = cashed
        self.giveback_10 = giveback

        return answer

    def update_01(self, trade_01):

        if trade_01 == 1:
            self.inventories[0] += self.last_requested_nb_coins_1 * self.last_answer_01 - self.last_cashed_01
            self.inventories[1] -= self.last_requested_nb_coins_1
            self.cash += self.last_cashed_01
        return True

    def update_10(self, trade_10):

        if trade_10 == 1:
            self.inventories[1] += self.last_requested_nb_coins_0 * self.last_answer_10 - self.giveback_10
            self.inventories[0] -= self.last_requested_nb_coins_0
            self.cash += self.last_cashed_10
        return True

    def mtm_value(self, swap_price_01):
        return self.cash + self.inventories[0] + swap_price_01 * self.inventories[1]

    def arb_01(self, swap_price_01, relative_cost, fixed_cost, step_ratio=50000, *args, **kwargs):
        if not self.support_arb:
            return 0
        return self._arb_01(swap_price_01, relative_cost, fixed_cost, step_ratio, *args, **kwargs)

    def arb_10(self, swap_price_10, relative_cost, fixed_cost, step_ratio=50000, *args, **kwargs):
        if not self.support_arb:
            return 0
        return self._arb_10(swap_price_10, relative_cost, fixed_cost, step_ratio, *args, **kwargs)

    def _arb_01(self, swap_price_01, relative_cost, fixed_cost, step_ratio, *args, **kwargs):

        last_price_01 = self.oracle.get()

        state = self.get_state()

        s = self.inventories[1] / step_ratio
        last_succesful_s = s
        amount = 0
        ko = 0
        ok = 0
        while ko < 4 and ok < step_ratio:
            proposed_swap_price_01 = self.proposed_swap_prices_01(s)
            if (proposed_swap_price_01 * (1 + relative_cost) > swap_price_01) or (self.unpeg_ratio is not None and proposed_swap_price_01 / last_price_01 >= self.unpeg_ratio):
                s /= 2
                ko += 1
            else:
                success = self.update_01(1)
                if not success:
                    break
                amount += s
                last_succesful_s = s
                ok += 1

        if amount:
            self.restore_state(state)
            proposed_swap_price_01 = self.proposed_swap_prices_01(amount)
            if proposed_swap_price_01 * (1 + relative_cost) < swap_price_01:
                self.update_01(1)
            elif ok > 1:
                proposed_swap_price_10 = self.proposed_swap_prices_10(amount - last_succesful_s)
                if proposed_swap_price_10 * (1 + relative_cost) < swap_price_01:
                    self.update_10(1)
                else:
                    amount = 0

        return amount

    def _arb_10(self, swap_price_10, relative_cost, fixed_cost, step_ratio, *args, **kwargs):
        
        last_price_10 = 1 / self.oracle.get()

        state = self.get_state()

        s = self.inventories[0] / step_ratio
        last_succesful_s = 0
        amount = 0
        ko = 0
        ok = 0
        while ko < 4 and ok < step_ratio:
            proposed_swap_price_10 = self.proposed_swap_prices_10(s)
            if (proposed_swap_price_10 * (1 + relative_cost) > swap_price_10) or (self.unpeg_ratio is not None and proposed_swap_price_10 / last_price_10 >= self.unpeg_ratio):
                s /= 2
                ko += 1
            else:
                success = self.update_10(1)
                if not success:
                    break
                amount += s
                last_succesful_s = s
                ok += 1

        if amount:
            self.restore_state(state)
            proposed_swap_price_10 = self.proposed_swap_prices_10(amount)
            if proposed_swap_price_10 * (1 + relative_cost) < swap_price_10:
                self.update_10(1)
            elif ok > 1:
                proposed_swap_price_10 = self.proposed_swap_prices_10(amount - last_succesful_s)
                if proposed_swap_price_10 * (1 + relative_cost) < swap_price_10:
                    self.update_10(1)
                else:
                    amount = 0
            
        return amount

    def restore_state(self, state):
        self.inventories = state["inventories"]
        self.cash = state["cash"]

    def get_state(self):
        return {
            "inventories": copy.deepcopy([float(v) for v in self.inventories]),
            "cash": float(self.cash),
        }

    def bot(self, *args, **kwargs):
        pass
    
    def is_pause(self):
        return self._pause

    def pause(self):
        self._pause = True
    
    def unpause(self):
        self._pause = False
