import numpy as np
from .logistictools import LogisticTools


class BaseLiquidityProvider:

    def __init__(self, name, initial_inventories, initial_cash, market, oracle):

        self.name = name
        self.initial_inventories = initial_inventories.copy()
        self.initial_cash = initial_cash
        self.inventories = initial_inventories
        self.cash = initial_cash
        self.oracle = oracle
        self.market = market
        self.last_requested_nb_coins_0 = None
        self.last_requested_nb_coins_1 = None
        self.last_answer_01 = None
        self.last_answer_10 = None
        self.last_cashed_01 = None
        self.last_cashed_10 = None

    def reset(self):

        self.inventories = self.initial_inventories.copy()
        self.cash = self.initial_cash
        self.oracle.reset()

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

    def update_10(self, trade_10):

        if trade_10 == 1:
            self.inventories[1] += self.last_requested_nb_coins_0 * self.last_answer_10
            self.inventories[0] -= self.last_requested_nb_coins_0 - self.last_cashed_10
            self.cash += self.last_cashed_10

    def mtm_value(self, swap_price_01):
        return self.cash + self.inventories[0] + swap_price_01 * self.inventories[1]


class LiquidityProviderCstDelta(BaseLiquidityProvider):

    def __init__(self, name, initial_inventories, initial_cash, market, oracle, delta):
        super().__init__(name, initial_inventories, initial_cash, market, oracle)
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


class LiquidityProviderAMMSqrt(BaseLiquidityProvider):

    def __init__(self, name, initial_inventories, initial_cash, market, oracle, delta):
        super().__init__(name, initial_inventories, initial_cash, market, oracle)
        self.delta = delta

    def pricing_function_01(self, nb_coins_1, swap_price_01):
        if self.inventories[1] <= nb_coins_1:
            return np.inf, 0.
        else:
            return self.get_cpmm_price(
                self.inventories[0], 0.5, self.inventories[1], 0.5, nb_coins_1, self.delta
            ), 0.

    def pricing_function_10(self, nb_coins_0, swap_price_10):
        if self.inventories[0] <= nb_coins_0:
            return np.inf, 0.
        else:
            return self.get_cpmm_price(
                self.inventories[1], 0.5, self.inventories[0], 0.5, nb_coins_0, self.delta
            ), 0.

    @staticmethod
    def get_cpmm_price(r_in, w_in, r_out, w_out, amount_out, delta):
        return (r_in * ((r_out / (r_out - amount_out)) ** (w_out / w_in) - 1)) / (amount_out * (1. - delta))


class LiquidityProviderBestClosedForm(BaseLiquidityProvider):

    def __init__(self, name, initial_inventories, initial_cash, market, oracle, gamma):

        super().__init__(name, initial_inventories, initial_cash, market, oracle)
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

        else:
            hodl_spread = (self.inventories[1] - self.initial_inventories[1]) * swap_price_01
            size = nb_coins_1 * swap_price_01
            z_index = np.argmin(np.abs(self.market.sizes-size))
            p = -2. * self.a * hodl_spread - self.b + size * self.a
            delta = self.lts_01[z_index].delta(p)
            return swap_price_01 * (1. + delta), delta * nb_coins_1 * swap_price_01

    def pricing_function_10(self, nb_coins_0, swap_price_10):

        if self.inventories[0] < nb_coins_0:
            return np.inf, 0.

        else:
            hodl_spread = (self.inventories[1] - self.initial_inventories[1]) / swap_price_10
            size = nb_coins_0
            z_index = np.argmin(np.abs(self.market.sizes-size))
            p = 2. * self.a * hodl_spread + self.b + size * self.a
            delta = self.lts_10[z_index].delta(p)

        return swap_price_10 * (1. + delta), delta * nb_coins_0


class LiquidityProviderSwaapV1(LiquidityProviderAMMSqrt):

    def __init__(
        self, name, initial_inventories, initial_cash, market, oracle, delta,
        z, horizon, lookback_calls, lookback_step,
    ):
        super().__init__(name, initial_inventories, initial_cash, market, oracle, delta)
        self.delta = delta
        self.initial_price = None
        self.initial_weights = {
            0: 0.5,
            1: 0.5
        }
        self.lookback_calls = lookback_calls
        self.lookback_step = lookback_step
        self.horizon = horizon
        self.z = z

    def set_init_price(self, time):
        self.initial_price = {
            1: self.oracle.get(time)
        }
        self.initial_price[0] = 1 / self.initial_price[1]

    def proposed_swap_prices_01(self, time, nb_coins_1):
        if self.initial_price is None:
            self.set_init_price(time)
        return super().proposed_swap_prices_01(time, nb_coins_1)

    def proposed_swap_prices_10(self, time, nb_coins_0):
        if self.initial_price is None:
            self.set_init_price(time)
        return super().proposed_swap_prices_10(time, nb_coins_0)

    def pricing_function_01(self, nb_coins_1, swap_price_01):
        return self.pricing_function_wrapper(
            nb_coins_1,
            swap_price_01,
            1
        )

    def pricing_function_10(self, nb_coins_0, swap_price_10):
        return self.pricing_function_wrapper(
            nb_coins_0,
            swap_price_10,
            0
        )

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
                return self.get_cpmm_price(
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
                a_price = self.get_cpmm_price(
                    r_in, dynamic_weights[index_in],
                    r_out, dynamic_weights[index_out],
                    a_amount,
                    delta
                )
                # shortage phase
                dynamic_weights = self.apply_spread(dynamic_weights, index_out, spread_factor)
                s_amount = amount_out - a_amount
                s_price = self.get_cpmm_price(
                    r_in + a_price * a_amount, dynamic_weights[index_in],
                    r_out - a_amount, dynamic_weights[index_out],
                    s_amount,
                    delta
                )
                return (a_price * a_amount + s_price * s_amount) / (a_amount + s_amount)

            # only shortage
            dynamic_weights = self.apply_spread(dynamic_weights, index_out, spread_factor)
            return self.get_cpmm_price(
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
        _weights [index_out] *= spread_factor
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
                    for i in range(0, n - 1)
                ]
            )
            - log_diff ** 2 / ts_diff
        ) / n

        # computes GBM quantile
        spread_factor = np.exp(mean * self.horizon + self.z * np.sqrt(2 * variance * self.horizon))
        return max(1, spread_factor)

    @staticmethod
    def get_in_reserve_at_price(r_in, w_in, r_out, w_out, price):
        r_in_eq = ((price * w_in / w_out * r_out) ** w_out) * r_in ** w_in
        return r_in_eq
