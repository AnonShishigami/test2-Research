import copy
from termios import VSTART

import numpy as np

from sandbox.strategies.base.strategy import BaseLiquidityProvider
from sandbox.strategies.curve_v2 import curve_v2_swap


class CurveV2(BaseLiquidityProvider):

    def __init__(
        self, name, initial_inventories, initial_cash, market, oracle, support_arb, initial_prices, dt_sim,
        A, gamma, 
        mid_fee, out_fee, allowed_extra_profit,
        fee_gamma, adjustment_step,
        admin_fee, ma_half_time,
        precisions=(1, 1, 1), input_precision_factor=curve_v2_swap.PRECISION
    ): 
        
        BaseLiquidityProvider.__init__(self, name, initial_inventories, initial_cash, market, oracle, support_arb)

        self.input_precision_factor = input_precision_factor
        self.initial_prices = [p * self.input_precision_factor if self.input_precision_factor != 1 else p for p in initial_prices]

        self.asset_0_index = 0
        self.asset_1_index = 1

        self.smart_contract = curve_v2_swap.Swap(
            A=A,
            gamma=gamma,
            mid_fee=mid_fee,
            out_fee=out_fee,
            allowed_extra_profit=allowed_extra_profit,
            fee_gamma=fee_gamma,
            adjustment_step=adjustment_step,
            admin_fee=admin_fee,
            ma_half_time=ma_half_time,
            initial_price=int(self.initial_prices[1] / self.initial_prices[0] * curve_v2_swap.PRECISION),
            precisions=precisions
        )
        self.smart_contract.add_liquidity(
            amounts=[
                initial_inventories[0] * self.input_precision_factor if self.input_precision_factor != 1 else initial_inventories[0], 
                initial_inventories[1] * self.input_precision_factor if self.input_precision_factor != 1 else initial_inventories[1], 
            ],
            min_mint_amount=0,
        )
        self.dt_sim = dt_sim

        self.smart_contract_state = None

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
        if self.initial_prices[self.asset_0_index] == 1:
            unscaled_amount = nb_coins_1
        else:
            unscaled_amount = nb_coins_1 * swap_price_01
        self.last_sell_amount_0 = int(unscaled_amount * self.input_precision_factor)
        dy, p = self.simulate_exchange(
            self.asset_0_index, self.asset_1_index, self.last_sell_amount_0,
        )
        self.last_requested_nb_coins_1 = dy / curve_v2_swap.PRECISION
        self.last_price = swap_price_01
        return p, 0

    def pricing_function_10(self, nb_coins_0, swap_price_10):
        if self.initial_prices[self.asset_0_index] == 1:
            unscaled_amount = nb_coins_0
        else:
            unscaled_amount = nb_coins_0 * swap_price_10
        self.last_sell_amount_1 = int(unscaled_amount * self.input_precision_factor)
        dy, p = self.simulate_exchange(
            self.asset_1_index, self.asset_0_index, self.last_sell_amount_1,
        )
        self.last_requested_nb_coins_0 = dy / curve_v2_swap.PRECISION
        self.last_price = 1 / swap_price_10
        return p, 0

    def _get_sc_state(self):
        return copy.deepcopy(self.smart_contract)

    def _restore_sc_state(self, state):
        self.smart_contract = copy.deepcopy(state)

    def get_state(self):
        return super().get_state() | {
            "sc": self._get_sc_state(),
        }

    def restore_state(self, state):
        super().restore_state(state)
        self._restore_sc_state(state["sc"])

    def proposed_swap_prices_01(self, time, nb_coins_1):
        self.smart_contract.block.set_timestamp(24 * 60 * 60 * time if self.dt_sim != 1 else time)
        return super().proposed_swap_prices_01(time, nb_coins_1)

    def proposed_swap_prices_10(self, time, nb_coins_0):
        self.smart_contract.block.set_timestamp(24 * 60 * 60 * time if self.dt_sim != 1 else time)
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
            if self.smart_contract_state:
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
            if self.smart_contract_state:
                self._restore_sc_state(self.smart_contract_state)
        return True

    def arb_01(self, time, swap_price_01, relative_cost, fixed_cost, step_ratio=1000):
            return super().arb_01(time, swap_price_01, relative_cost, fixed_cost, step_ratio=step_ratio)

    def arb_10(self, time, swap_price_10, relative_cost, fixed_cost, step_ratio=1000):
            return super().arb_10(time, swap_price_10, relative_cost, fixed_cost, step_ratio=step_ratio)

