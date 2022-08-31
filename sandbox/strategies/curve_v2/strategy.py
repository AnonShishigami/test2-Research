import copy

import numpy as np

from sandbox.strategies.base.strategy import BaseLiquidityProvider
from sandbox.strategies.curve_v2 import curve_v2_swap


class CurveV2(BaseLiquidityProvider):

    def __init__(
        self, name, initial_inventories, initial_cash, market, oracle, support_arb, initial_prices, dt_sim,
        A=5400000, gamma=20000000000000, precisions=(1, 1, 1), input_precision_factor=curve_v2_swap.PRECISION
    ): 
        
        BaseLiquidityProvider.__init__(self, name, initial_inventories, initial_cash, market, oracle, support_arb)

        self.input_precision_factor = input_precision_factor
        self.initial_prices = [p * self.input_precision_factor for p in [1] + initial_prices]

        self.asset_0_index = 1
        self.asset_1_index = 2

        self.smart_contract = curve_v2_swap.Swap(
            A=A,
            gamma=gamma,
            mid_fee=5000000,
            out_fee=30000000,
            allowed_extra_profit=200000000000,
            fee_gamma=500000000000000,
            adjustment_step=500000000000000,
            admin_fee=5000000000,
            ma_half_time=600,
            initial_prices=self.initial_prices[1:],
            initial_balances=[
                initial_inventories[0] * self.initial_prices[1], 
                initial_inventories[0] * self.input_precision_factor, 
                initial_inventories[1] * self.input_precision_factor
            ],
            precisions=precisions
        )
        self.dt_sim = dt_sim

        self._tmp_sc_state = None

    def process_exchange(self, sell_idx, buy_idx, sell_amount):
        self.smart_contract_state = self._get_sc_state()
        dy = self.smart_contract.exchange(
            sell_idx, buy_idx, sell_amount, 0,
        )
        return dy

    def simulate_exchange(self, sell_idx, buy_idx, sell_amount):
        try:
            dy = self.smart_contract.get_dy(
                sell_idx, buy_idx, sell_amount
            )
            return dy, sell_amount / dy
        except Exception as e:
            return 0, np.inf

    def pricing_function_01(self, nb_coins_1, swap_price_01):
        unscaled_amount = nb_coins_1 * self.initial_prices[self.asset_1_index] / self.initial_prices[self.asset_0_index]
        self.last_sell_amount_0 = int(unscaled_amount * self.input_precision_factor)
        dy, p = self.simulate_exchange(
            self.asset_0_index, self.asset_1_index, self.last_sell_amount_0,
        )
        self.last_requested_nb_coins_1 = dy / curve_v2_swap.PRECISION
        return p, 0

    def pricing_function_10(self, nb_coins_0, swap_price_10):
        unscaled_amount = nb_coins_0 * self.initial_prices[self.asset_0_index] / self.initial_prices[self.asset_1_index]
        self.last_sell_amount_1 = int(unscaled_amount * self.input_precision_factor)
        dy, p = self.simulate_exchange(
            self.asset_1_index, self.asset_0_index, self.last_sell_amount_1,
        )
        self.last_requested_nb_coins_0 = dy / curve_v2_swap.PRECISION
        return p, 0

    def _get_sc_state(self):
        return copy.deepcopy(self.smart_contract.__dict__)

    def _restore_sc_state(self, state):
        self.smart_contract.__dict__ = copy.deepcopy(state)

    def get_state(self):
        return {
            "inventories": [float(v) for v in self.inventories],
            "cash": float(self.cash),
            "sc": self._get_sc_state(),
        }

    def restore_state(self, state):
        self.inventories = state["inventories"]
        self.cash = state["cash"]
        self._restore_sc_state(state["sc"])

    def proposed_swap_prices_01(self, time, nb_coins_1):
        self.smart_contract.block.set_timestamp(time / self.dt_sim)
        return super().proposed_swap_prices_01(time, nb_coins_1)

    def proposed_swap_prices_10(self, time, nb_coins_0):
        self.smart_contract.block.set_timestamp(time / self.dt_sim)
        return super().proposed_swap_prices_10(time, nb_coins_0)
    
    def update_01(self, trade_01):
        if trade_01 == 1:
            try:
                dy = self.process_exchange(self.asset_0_index, self.asset_1_index, self.last_sell_amount_0) / curve_v2_swap.PRECISION
                assert dy == self.last_requested_nb_coins_1
                self.inventories[0] += self.last_requested_nb_coins_1 * self.last_answer_01 - self.last_cashed_01
                self.inventories[1] -= self.last_requested_nb_coins_1
                self.cash += self.last_cashed_01
            except Exception as e:
                self._restore_sc_state(self.smart_contract_state)
                return False
        else:
            self._restore_sc_state(self.smart_contract_state)
        return True

    def update_10(self, trade_10):
        if trade_10 == 1:
            try:
                dy = self.process_exchange(self.asset_1_index, self.asset_0_index, self.last_sell_amount_1) / curve_v2_swap.PRECISION
                assert dy == self.last_requested_nb_coins_0
                self.inventories[1] += self.last_requested_nb_coins_0 * self.last_answer_10
                self.inventories[0] -= self.last_requested_nb_coins_0 - self.last_cashed_10
                self.cash += self.last_cashed_10
            except Exception as e:
                self._restore_sc_state(self.smart_contract_state)
                return False
        else:
            self._restore_sc_state(self.smart_contract_state)
        return True

    def arb_01(self, time, swap_price_01, relative_cost, fixed_cost, step_ratio=10000):
            return super().arb_01(time, swap_price_01, relative_cost, fixed_cost, step_ratio=1000)

    def arb_10(self, time, swap_price_10, relative_cost, fixed_cost, step_ratio=10000):
            return super().arb_10(time, swap_price_10, relative_cost, fixed_cost, step_ratio=1000)

