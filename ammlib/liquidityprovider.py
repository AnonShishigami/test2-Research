import copy

import numpy as np
from .control_tools import LogisticExtended, MixedLogisticsExtended
from .demand_curve import Logistic, MixedLogistics
from .logistictools import LogisticTools
from .curve_v2 import curve_v2_swap


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

    def arb_01(self, time, swap_price_01, relative_cost, fixed_cost):

        if not self.support_arb:
            return 0

        state = self._get_state()

        s = self.inventories[1] / 10000
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
            self.proposed_swap_prices_01(time, amount)
            self.update_01(1)

        return amount

    def arb_10(self, time, swap_price_10, relative_cost, fixed_cost):

        if not self.support_arb:
            return 0

        state = self._get_state()

        s = self.inventories[0] / 10000
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
            self.proposed_swap_prices_10(time, amount)
            self.update_10(1)

        return amount

    def restore_state(self, state):
        self.inventories = state["inventories"]
        self.cash = state["cash"]

    def _get_state(self):
        return {
            "inventories": [float(v) for v in self.inventories],
            "cash": float(self.cash),
        }


class LiquidityProviderCstDelta(BaseLiquidityProvider):

    def __init__(self, name, initial_inventories, initial_cash, market, oracle, support_arb, delta):
        super().__init__(name, initial_inventories, initial_cash, market, oracle, support_arb)
        self.delta = delta

    def pricing_function_01(self, nb_coins_1, swap_price_01):
        if self.inventories[1] < nb_coins_1:
            return np.inf, 0.
        else:
            return swap_price_01 * (1. + self.delta), self.delta * nb_coins_1 * swap_price_01

    def pricing_function_10(self, nb_coins_0, swap_price_10):
        if self.inventories[0] < nb_coins_0:
            return np.inf, 0.
        else:
            return swap_price_10 * (1. + self.delta), self.delta * nb_coins_0

    def arb_01(self, time, swap_price_01, relative_cost, fixed_cost):
        if not self.support_arb:
            return 0
        if self.oracle.get(time) * (1 + relative_cost) > swap_price_01:
            return 0
        amount = self.inventories[1]
        self.proposed_swap_prices_01(time, amount)
        self.update_01(1)
        return amount

    def arb_10(self, time, swap_price_10, relative_cost, fixed_cost):
        if not self.support_arb:
            return 0
        if 1. / self.oracle.get(time) * (1 + relative_cost) > swap_price_10:
            return 0
        amount = self.inventories[0]
        self.proposed_swap_prices_10(time, amount)
        self.update_10(1)
        return amount


class LiquidityProviderCFMMPowers(BaseLiquidityProvider):

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
                self.inventories[0], self.w0, self.inventories[1], self.w1, nb_coins_1, self.delta
            ), 0.

    def pricing_function_10(self, nb_coins_0, swap_price_10):
        if self.inventories[0] <= nb_coins_0:
            return np.inf, 0.
        else:
            return self.get_cfmm_powers_price(
                self.inventories[1], self.w1, self.inventories[0], self.w0, nb_coins_0, self.delta
            ), 0.

    @staticmethod
    def get_cfmm_powers_price(r_in, w_in, r_out, w_out, amount_out, delta):
        return (r_in * ((r_out / (r_out - amount_out)) ** (w_out / w_in) - 1)) / (amount_out * (1. - delta))

class LiquidityProviderCFMMSqrt(LiquidityProviderCFMMPowers):

    def __init__(self, name, initial_inventories, initial_cash, market, oracle, support_arb, delta):
        super().__init__(name, initial_inventories, initial_cash, market, oracle, support_arb, np.array([0.5, 0.5]), delta)


class LiquidityProviderConcentratedCFMMSqrt(LiquidityProviderCFMMSqrt):

    @staticmethod
    def get_cfmm_powers_price(r_in, w_in, r_out, w_out, amount_out, delta, concentration_factor=100):
        r_in *= concentration_factor
        r_out *= concentration_factor
        return (r_in * ((r_out / (r_out - amount_out)) ** (w_out / w_in) - 1)) / (amount_out * (1. - delta))


class LiquidityProviderCFMMSqrtCloseArb(LiquidityProviderCFMMSqrt):

    def arb_01(self, time, swap_price_01, relative_cost, fixed_cost):
        if not self.support_arb:
            return 0
        amount = self.inventories[1] - np.sqrt(
            self.inventories[0] * self.inventories[1] / (swap_price_01 / (1 + relative_cost) * (1 - self.delta))
        )
        if amount > 0:
            self.proposed_swap_prices_01(time, amount)
            self.update_01(1)
        return amount

    def arb_10(self, time, swap_price_10, relative_cost, fixed_cost):
        if not self.support_arb:
            return 0
        amount = self.inventories[0] - np.sqrt(
            self.inventories[0] * self.inventories[1] / (swap_price_10 / (1 + relative_cost) * (1 - self.delta))
        )
        if amount > 0:
            self.proposed_swap_prices_10(time, amount)
            self.update_10(1)
        return amount


class LiquidityProviderBestClosedForm(BaseLiquidityProvider):

    def __init__(self, name, initial_inventories, initial_cash, market, oracle, support_arb, gamma):

        super().__init__(name, initial_inventories, initial_cash, market, oracle, support_arb)
        self.gamma = gamma

        self.lts_01 = [LogisticTools(obj.lambda_, obj.a, obj.b) for obj in self.market.intensities_functions_01_object]
        self.lts_10 = [LogisticTools(obj.lambda_, obj.a, obj.b) for obj in self.market.intensities_functions_10_object]

        plus21 = np.sum(np.array([z * (self.lts_01[k].H_second(0.) + self.lts_10[k].H_second(0.))  for k,z in enumerate(self.market.sizes)]))
        minus22 = np.sum(np.array([z**2 * (self.lts_01[k].H_second(0.) - self.lts_10[k].H_second(0.)) for k, z in enumerate(self.market.sizes)]))
        minus11 = np.sum(np.array([z * (self.lts_01[k].H_prime(0.) - self.lts_10[k].H_prime(0.)) for k, z in enumerate(self.market.sizes)]))

        self.a = (self.market.sigma**2 + np.sqrt(self.market.sigma**4 + 4. * self.gamma * self.market.sigma**2 * plus21)) / (4. * plus21)
        self.b = - (minus11 - minus22 * self.a) / plus21

    def pricing_function_01(self, nb_coins_1, swap_price_01):

        if self.inventories[1] < nb_coins_1:
            return np.inf, 0.

        hodl_spread = (self.inventories[1] - self.initial_inventories[1]) * swap_price_01
        size = nb_coins_1 * swap_price_01
        z_index = np.argmin(np.abs(self.market.sizes-size))
        p = -2. * self.a * hodl_spread - self.b + size * self.a
        delta = self.lts_01[z_index].delta(p)
        return swap_price_01 * (1. + delta), delta * nb_coins_1 * swap_price_01

    def pricing_function_10(self, nb_coins_0, swap_price_10):

        if self.inventories[0] < nb_coins_0:
            return np.inf, 0.

        hodl_spread = (self.inventories[1] - self.initial_inventories[1]) / swap_price_10
        size = nb_coins_0
        z_index = np.argmin(np.abs(self.market.sizes-size))
        p = 2. * self.a * hodl_spread + self.b + size * self.a
        delta = self.lts_10[z_index].delta(p)
        return swap_price_10 * (1. + delta), delta * nb_coins_0


class LiquidityProviderSwaapV1(LiquidityProviderCFMMPowers):

    def __init__(
        self, name, initial_inventories, initial_cash, market, oracle, support_arb, delta,
        z, horizon, lookback_calls, lookback_step,
    ):
        super().__init__(name, initial_inventories, initial_cash, market, oracle, support_arb, np.array([0.5, 0.5]), delta)

        self.initial_price = None
        self.initial_weights = {
            0: self.w0,
            1: self.w1
        }
        self.lookback_calls = lookback_calls
        self.lookback_step = lookback_step
        self.horizon = horizon
        self.z = z

    def set__init__price(self, time):
        self.initial_price = {
            1: self.oracle.get(time)
        }
        self.initial_price[0] = 1. / self.initial_price[1]

    def proposed_swap_prices_01(self, time, nb_coins_1):
        if self.initial_price is None:
            self.set__init__price(time)
        return super().proposed_swap_prices_01(time, nb_coins_1)

    def proposed_swap_prices_10(self, time, nb_coins_0):
        if self.initial_price is None:
            self.set__init__price(time)
        return super().proposed_swap_prices_10(time, nb_coins_0)

    def pricing_function_01(self, nb_coins_1, swap_price_01):
        return self.pricing_function_wrapper(
            nb_coins_1,
            swap_price_01,
            1
        )

    def pricing_function_10(self, nb_coins_0, swap_price_10):
        p, cash = self.pricing_function_wrapper(
            nb_coins_0,
            swap_price_10,
            0
        )
        return p, cash

    def pricing_function_wrapper(self, amount_out, price, index_out):
        index_in = 0 if index_out == 1 else 1
        dynamic_weights = self.get_dynamic_weights(price, index_out)
        price = self.pricing_function(
            self.inventories[index_in],
            index_in,
            self.inventories[index_out],
            index_out,
            dynamic_weights,
            amount_out,
            price,
            self.delta
        )
        return price, 0.

    def pricing_function(self, r_in, index_in, r_out, index_out, dynamic_weights, amount_out, price_out, delta):
        if r_out <= amount_out:
            return np.inf, 0.
        else:
            # weights dynamically adjusted according to assets's relative performance

            # defines abundance / shortage boundary
            r_out_at_oracle_price = self.get_in_reserve_at_price(
                r_out, dynamic_weights[index_out],
                r_in, dynamic_weights[index_in],
                1 / price_out
            )
            if r_out - amount_out > r_out_at_oracle_price:
                # abundance phase only
                return self.get_cfmm_powers_price(
                    r_in, dynamic_weights[index_in],
                    r_out, dynamic_weights[index_out],
                    amount_out,
                    delta
                )

            # will experience shortage phase
            spread_factor = self.get_spread_factor(route=f"{index_in}{index_out}")

            if r_out > r_out_at_oracle_price:
                # abundance before shortage phase
                a_amount = r_out - r_out_at_oracle_price
                a_price = self.get_cfmm_powers_price(
                    r_in, dynamic_weights[index_in],
                    r_out, dynamic_weights[index_out],
                    a_amount,
                    delta
                )
                # shortage phase
                dynamic_weights = self.apply_spread(dynamic_weights, index_out, spread_factor)
                s_amount = amount_out - a_amount
                s_price = self.get_cfmm_powers_price(
                    r_in + a_price * a_amount, dynamic_weights[index_in],
                    r_out - a_amount, dynamic_weights[index_out],
                    s_amount,
                    delta
                )
                return (a_price * a_amount + s_price * s_amount) / (a_amount + s_amount)

            # only shortage
            dynamic_weights = self.apply_spread(dynamic_weights, index_out, spread_factor)
            return self.get_cfmm_powers_price(
                r_in, dynamic_weights[index_in],
                r_out, dynamic_weights[index_out],
                amount_out,
                delta
            )

    def get_dynamic_weights(self, running_price, index):
        weights = dict((k, self.initial_weights[k]) for k in self.initial_weights)
        weights[index] *= running_price / self.initial_price[index]
        return self.normalize_weights(weights)

    def apply_spread(self, weights, index_out, spread_factor):
        _weights = dict((k, weights[k]) for k in weights)
        _weights[index_out] *= spread_factor
        return self.normalize_weights(_weights)

    @staticmethod
    def normalize_weights(weights):
        w = np.sum(list(weights.values()))
        return dict((k, weights[k] / w) for k in weights)

    # GBM spread computation
    def get_spread_factor(self, route):

        # gets historical prices
        seq = self.oracle.get_last_timestamped_prices(
            lookback_calls=self.lookback_calls,
            route=route,
            lookback_step=self.lookback_step
        )
        if len(seq) < 2:
            return 1
        n = len(seq)
        log_diff = np.log(seq[0]["price"] / seq[-1]["price"])
        ts_diff = seq[0]["ts"] - seq[-1]["ts"]

        # computes GBM MLE
        mean = log_diff / ts_diff
        variance = (
            np.sum(
                [
                    np.log(seq[i]["price"] / seq[i + 1]["price"]) ** 2 / (seq[i]["ts"] - seq[i + 1]["ts"])
                    for i in range(n - 1)
                ]
            )
            - log_diff ** 2 / ts_diff
        ) / n

        # computes GBM quantile
        spread_factor = np.exp(mean * self.horizon + self.z * np.sqrt(2. * variance * self.horizon))
        return max(1, spread_factor)

    @staticmethod
    def get_in_reserve_at_price(r_in, w_in, r_out, w_out, price):
        r_in_eq = ((price * w_in / w_out * r_out) ** w_out) * r_in ** w_in
        return r_in_eq


class LiquidityProviderCurveV2(BaseLiquidityProvider):

    def __init__(self, name, initial_inventories, initial_cash, market, oracle, support_arb, initial_prices):

        BaseLiquidityProvider.__init__(self, name, initial_inventories, initial_cash, market, oracle, support_arb)

        self.initial_prices = [p * curve_v2_swap.PRECISION for p in [1] + initial_prices]

        self.asset_0_index = 1
        self.asset_1_index = 2

        self.smart_contract = curve_v2_swap.Swap(
            owner="",
            admin_fee_receiver="",
            A=5400000,
            gamma=20000000000000,
            mid_fee=5000000,
            out_fee=30000000,
            allowed_extra_profit=200000000000,
            fee_gamma=500000000000000,
            adjustment_step=500000000000000,
            admin_fee=5000000000,
            ma_half_time=600,
            initial_prices=self.initial_prices[1:],
            initial_balances=[initial_inventories[0] * self.initial_prices[1], initial_inventories[0]*curve_v2_swap.PRECISION, initial_inventories[1]*curve_v2_swap.PRECISION],
        )

        self._tmp_sc_state = None

    def pricing_function_01(self, nb_coins_1, swap_price_01):
        amount = nb_coins_1 * self.initial_prices[self.asset_1_index] / self.initial_prices[self.asset_0_index]
        dy, A_gamma, xp, ix, p, balances, D, future_A_gamma_time = self.smart_contract.exchange(
            self.asset_0_index, self.asset_1_index, amount * curve_v2_swap.PRECISION, 0,
        )
        dy /= curve_v2_swap.PRECISION
        self.last_requested_nb_coins_1 = dy

        self._tmp_sc_state = {}
        self._tmp_sc_state["tweak_price_parameters"] = (A_gamma, xp, ix, p, 0)
        self._tmp_sc_state["balances"] = balances
        self._tmp_sc_state["D"] = D
        self._tmp_sc_state["future_A_gamma_time"] = future_A_gamma_time

        p = amount / dy

        return p, 0

    def pricing_function_10(self, nb_coins_0, swap_price_10):
        amount = nb_coins_0 * self.initial_prices[self.asset_0_index] / self.initial_prices[self.asset_1_index]
        dy, A_gamma, xp, ix, p, balances, D, future_A_gamma_time = self.smart_contract.exchange(
            self.asset_1_index, self.asset_0_index, amount * curve_v2_swap.PRECISION, 0,
        )
        dy /= curve_v2_swap.PRECISION
        self.last_requested_nb_coins_0 = dy

        self._tmp_sc_state = {}
        self._tmp_sc_state["tweak_price_parameters"] = (A_gamma, xp, ix, p, 0)
        self._tmp_sc_state["balances"] = balances
        self._tmp_sc_state["D"] = D
        self._tmp_sc_state["future_A_gamma_time"] = future_A_gamma_time
        
        p = amount / dy
        
        return p, 0

    def update_01(self, trade_01):
        if trade_01:
            success = self.update_callabck()
            if not success:
                return False
        return super().update_01(trade_01)

    def update_10(self, trade_10):
        if trade_10:
            success = self.update_callabck()
            if not success:
                return False
        return super().update_10(trade_10)

    def update_callabck(self):
        return self._update_callabck(
            self._tmp_sc_state["tweak_price_parameters"],
            self._tmp_sc_state["balances"],
            self._tmp_sc_state["D"],
            self._tmp_sc_state["future_A_gamma_time"],
        )

    def _update_callabck(self, tweak_price_parameters, balances, D, future_A_gamma_time):
        state = self._get_state()
        try:
            self.smart_contract.set_balances(balances)
            self.smart_contract.set_D(D)
            self.smart_contract.set_future_A_gamma_time(future_A_gamma_time)
            self.smart_contract.tweak_price(*tweak_price_parameters)
        except Exception as e:
            self.restore_state(state)
            # sanity check
            if abs(self.smart_contract.get_balances()[1] / 10**18 - self.inventories[0]) > 1e-8 or abs(self.smart_contract.get_balances()[2] / 10**18 - self.inventories[1]) > 1e-8:
                print("init state:", state)
                sys.exit()
            return False
        return True

    def _get_state(self):
        return {
            "inventories": [float(v) for v in self.inventories],
            "cash": float(self.cash),
            "sc": copy.deepcopy(self.smart_contract),
        }

    def restore_state(self, state):
        self.inventories = state["inventories"]
        self.cash = state["cash"]
        self.smart_contract = state["sc"]
