import numpy as np

from sandbox.strategies.cfmm_sqrt.strategy import CFMMSqrt


class SwaapV1(CFMMSqrt):

    def __init__(
        self, name, initial_inventories, initial_cash, market, oracle, support_arb, delta,
        z, horizon, lookback_calls, lookback_step,
        concentration=1
    ):
        super().__init__(name, initial_inventories, initial_cash, market, oracle, support_arb, [1, 1], delta, concentration=concentration)

        self.initial_price = None
        self.initial_weights = {
            0: self.w0 / (self.w0 + self.w1),
            1: self.w1 / (self.w0 + self.w1),
        }
        
        self.lookback_calls = lookback_calls
        self.lookback_step = lookback_step
        self.horizon = horizon
        self.z = z
        self.concentrated_inventories = [i * np.sqrt(self.concentration) for i in self.inventories]

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
            0,
        )
        return p, cash

    def pricing_function_wrapper(self, amount_out, price, index_out):
        if min(self.concentrated_inventories[index_out], self.inventories[index_out]) < amount_out:
            return np.inf, 0.
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

            eq_r_in = r_in
            eq_r_out = r_out

            r_in = self.concentrated_inventories[index_in]
            r_out = self.concentrated_inventories[index_out]
        
            # defines abundance / shortage boundary
            r_out_at_oracle_price = self.get_in_reserve_at_price(
                eq_r_out, dynamic_weights[index_out],
                eq_r_in, dynamic_weights[index_in],
                1 / price_out
            )

            if eq_r_out - amount_out > r_out_at_oracle_price:
                # abundance phase only
                return self.get_cfmm_powers_price(
                    r_in, dynamic_weights[index_in],
                    r_out, dynamic_weights[index_out],
                    amount_out,
                    delta,
                )

            # will experience shortage phase
            spread_factor = self.get_spread_factor(route=f"{index_in}{index_out}")

            # transposing to the concentrated liquidity domain
            r_out_at_oracle_price = r_out - (eq_r_out - r_out_at_oracle_price)

            if r_out > r_out_at_oracle_price:
                # abundance before shortage phase
                a_amount = r_out - r_out_at_oracle_price
                a_price = self.get_cfmm_powers_price(
                    r_in, dynamic_weights[index_in],
                    r_out, dynamic_weights[index_out],
                    a_amount,
                    delta,
                )
                # shortage phase
                dynamic_weights = self.apply_spread(dynamic_weights, index_out, spread_factor)
                s_amount = amount_out - a_amount
                s_price = self.get_cfmm_powers_price(
                    r_in + a_price * a_amount, dynamic_weights[index_in],
                    r_out - a_amount, dynamic_weights[index_out],
                    s_amount,
                    delta,
                )
                return (a_price * a_amount + s_price * s_amount) / (a_amount + s_amount)

            # only shortage
            dynamic_weights = self.apply_spread(dynamic_weights, index_out, spread_factor)
            return self.get_cfmm_powers_price(
                r_in, dynamic_weights[index_in],
                r_out, dynamic_weights[index_out],
                amount_out,
                delta,
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
    