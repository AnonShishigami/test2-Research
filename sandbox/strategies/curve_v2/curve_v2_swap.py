# @version 0.3.1
# (c) Curve.Fi, 2021
# Pool for USDT/BTC/ETH or similar

import time

import numpy as np

from sandbox.strategies.curve_v2.vyper_utils import shift, bitwise_or, bitwise_and, empty, call_by_value, shallow_array_copy
import sandbox.strategies.curve_v2.curve_v2_math as curve_v2_math
from sandbox.strategies.curve_v2.curve_v2_config import (
    N_COINS,
    PRECISION,
    KILL_DEADLINE_DT,
    NOISE_FEE
)


class Block:
    
    def __init__(self):
        self._timestamp = 0
    
    @property
    def timestamp(self):
        return self._timestamp

    def update_timestamp(self, step=2):
        self._timestamp += step

    def set_timestamp(self, timestamp):
        self._timestamp = int(timestamp)


class Swap:

    @call_by_value
    def __init__(
        self,
        A: int,
        gamma: int,
        mid_fee: int,
        out_fee: int,
        allowed_extra_profit: int,
        fee_gamma: int,
        adjustment_step: int,
        admin_fee: int,
        ma_half_time: int,
        initial_price: int, 
        precisions: list,
    ):

        # Pack A and gamma:
        # shifted A + gamma
        A_gamma: int = shift(A, 128)
        A_gamma = bitwise_or(A_gamma, gamma)
        self.initial_A_gamma = A_gamma
        self.future_A_gamma = A_gamma
        self.initial_A_gamma_time = 0

        self.block = Block()

        self.mid_fee = mid_fee
        self.out_fee = out_fee
        self.allowed_extra_profit = allowed_extra_profit
        self.fee_gamma = fee_gamma
        self.adjustment_step = adjustment_step
        self.admin_fee = admin_fee
        self.precisions = precisions

        self.price_scale = initial_price
        self.price_oracle = initial_price
        self.last_prices = initial_price
        self.last_prices_timestamp = self.block.timestamp
        self.ma_half_time = ma_half_time

        self.xcp_profit_a = 10**18

        self.kill_deadline = self.block.timestamp + KILL_DEADLINE_DT

        # hardcoded
        self.future_A_gamma_time = 0
        self.D = 0
        self.total_supply = 0
        self.xcp_profit = 0
        self.virtual_price = 0
        self.not_adjusted = False
        self.is_killed = False
        self.balances = [0, 0]
    
    def _A_gamma(self) -> list:
        t1: int = self.future_A_gamma_time

        A_gamma_1: int = self.future_A_gamma
        gamma1: int = bitwise_and(A_gamma_1, 2**128-1)
        A1: int = shift(A_gamma_1, -128)

        if self.block.timestamp < t1:
            # handle ramping up and down of A
            A_gamma_0: int = self.initial_A_gamma
            t0: int = self.initial_A_gamma_time

            # Less readable but more compact way of writing and converting to int
            # gamma0: int = bitwise_and(A_gamma_0, 2**128-1)
            # A0: int = shift(A_gamma_0, -128)
            # A1 = A0 + (A1 - A0) * (self.block.timestamp - t0) // (t1 - t0)
            # gamma1 = gamma0 + (gamma1 - gamma0) * (self.block.timestamp - t0) // (t1 - t0)

            t1 -= t0
            t0 = self.block.timestamp - t0
            t2: int = t1 - t0

            A1 = (shift(A_gamma_0, -128) * t2 + A1 * t0) // t1
            gamma1 = (bitwise_and(A_gamma_0, 2**128-1) * t2 + gamma1 * t0) // t1

        return [A1, gamma1]

    def A(self) -> int:
        return self._A_gamma()[0]

    def gamma(self) -> int:
        return self._A_gamma()[1]

    def _fee(self, xp: list) -> int:
        """
        f = fee_gamma // (fee_gamma + (1 - K))
        where
        K = prod(x) // (sum(x) // N)**N
        (all normalized to 1e18)
        """
        fee_gamma: int = self.fee_gamma
        f: int = xp[0] + xp[1]  # sum
        f = fee_gamma * 10**18 // (
            fee_gamma + 10**18 - (10**18 * N_COINS**N_COINS) * xp[0] // f * xp[1] // f
        )
        return (self.mid_fee * f + self.out_fee * (10**18 - f)) // 10**18
        # return 1 / 100 * 10**10

    def fee_calc(self, xp: list) -> int:
        return self._fee(xp)

    def get_xcp(self, D: int) -> int:
        x: list = [D // N_COINS, D * PRECISION // (self.price_scale * N_COINS)]
        return curve_v2_math.geometric_mean(x, True)

    @call_by_value
    def tweak_price(self, A_gamma: list,_xp: list, p_i: int, new_D: int):
        price_oracle: int = self.price_oracle
        last_prices: int = self.last_prices
        price_scale: int = self.price_scale
        last_prices_timestamp: int = self.last_prices_timestamp
        p_new: int = 0

        if last_prices_timestamp < self.block.timestamp:
            # MA update required
            ma_half_time: int = self.ma_half_time
            alpha: int = curve_v2_math.halfpow((self.block.timestamp - last_prices_timestamp) * 10**18 // ma_half_time)
            price_oracle = (last_prices * (10**18 - alpha) + price_oracle * alpha) // 10**18
            self.price_oracle = price_oracle
            self.last_prices_timestamp = self.block.timestamp

        D_unadjusted: int = new_D  # Withdrawal methods know new D already
        if new_D == 0:
            # We will need this a few times (35k gas)
            D_unadjusted = curve_v2_math.newton_D(A_gamma[0], A_gamma[1], _xp)

        if p_i > 0:
            last_prices = p_i

        else:
            # calculate real prices
            __xp: list = shallow_array_copy(_xp)
            dx_price: int = __xp[0] // 10**6
            __xp[0] += dx_price
            last_prices = price_scale * dx_price // (_xp[1] - curve_v2_math.newton_y(A_gamma[0], A_gamma[1], __xp, D_unadjusted, 1))

        self.last_prices = last_prices

        total_supply: int = self.total_supply
        old_xcp_profit: int = self.xcp_profit
        old_virtual_price: int = self.virtual_price

        # Update profit numbers without price adjustment first
        xp: list = [D_unadjusted // N_COINS, D_unadjusted * PRECISION // (N_COINS * price_scale)]
        xcp_profit: int = 10**18
        virtual_price: int = 10**18

        if old_virtual_price > 0:
            xcp: int = curve_v2_math.geometric_mean(xp, True)
            virtual_price = 10**18 * xcp // total_supply
            xcp_profit = old_xcp_profit * virtual_price // old_virtual_price

            t: int = self.future_A_gamma_time
            if virtual_price < old_virtual_price and t == 0:
                raise Exception("Loss")
            if t == 1:
                self.future_A_gamma_time = 0

        self.xcp_profit = xcp_profit

        needs_adjustment: bool = self.not_adjusted
        # if not needs_adjustment and (virtual_price-10**18 > (xcp_profit-10**18)/2 + self.allowed_extra_profit):
        # (re-arrange for gas efficiency)
        if not needs_adjustment and (virtual_price * 2 - 10**18 > xcp_profit + 2*self.allowed_extra_profit):
            needs_adjustment = True
            self.not_adjusted = True

        if needs_adjustment:
            adjustment_step: int = self.adjustment_step
            norm: int = price_oracle * 10**18 // price_scale
            if norm > 10**18:
                norm -= 10**18
            else:
                norm = 10**18 - norm

            if norm > adjustment_step and old_virtual_price > 0:
                p_new = (price_scale * (norm - adjustment_step) + adjustment_step * price_oracle) // norm

                # Calculate balances*prices
                xp = [_xp[0], _xp[1] * p_new // price_scale]

                # Calculate "extended constant product" invariant xCP and virtual price
                D: int = curve_v2_math.newton_D(A_gamma[0], A_gamma[1], xp)
                xp = [D // N_COINS, D * PRECISION // (N_COINS * p_new)]
                # We reuse old_virtual_price here but it's not old anymore
                old_virtual_price = 10**18 * curve_v2_math.geometric_mean(xp, True) // total_supply

                # Proceed if we've got enough profit
                # if (old_virtual_price > 10**18) and (2 * (old_virtual_price - 10**18) > xcp_profit - 10**18):
                if (old_virtual_price > 10**18) and (2 * old_virtual_price - 10**18 > xcp_profit):
                    self.price_scale = p_new
                    self.D = D
                    self.virtual_price = old_virtual_price

                    return

                else:
                    self.not_adjusted = False

                    # Can instead do another flag variable if we want to save bytespace
                    self.D = D_unadjusted
                    self.virtual_price = virtual_price
                    # self._claim_admin_fees()

                    return

        # If we are here, the price_scale adjustment did not happen
        # Still need to update the profit counter and D
        self.D = D_unadjusted
        self.virtual_price = virtual_price

    @call_by_value
    def exchange(self, i: int, j: int, dx: int, min_dy: int) -> int:
        assert not self.is_killed  # dev: the pool is killed
        assert i != j  # dev: coin index out of range
        assert i < N_COINS  # dev: coin index out of range
        assert j < N_COINS  # dev: coin index out of range
        assert dx > 0  # dev: do not exchange 0 coins

        A_gamma: list = self._A_gamma()
        xp: int[N_COINS] = shallow_array_copy(self.balances)
        p: int = 0
        dy: int = 0

        y: int = xp[j]
        x0: int = xp[i]
        xp[i] = x0 + dx
        self.balances[i] = xp[i]

        price_scale: int = self.price_scale

        xp = [xp[0] * self.precisions[0], xp[1] * price_scale * self.precisions[1] // PRECISION]

        prec_i: int = self.precisions[0]
        prec_j: int = self.precisions[1]
        if i == 1:
            prec_i = self.precisions[1]
            prec_j = self.precisions[0]

        # In case ramp is happening
        t: int = self.future_A_gamma_time
        if t > 0:
            x0 *= prec_i
            if i > 0:
                x0 = x0 * price_scale // PRECISION
            x1: int = xp[i]  # Back up old value in xp
            xp[i] = x0
            self.D = curve_v2_math.newton_D(A_gamma[0], A_gamma[1], xp)
            xp[i] = x1  # And restore
            if self.block.timestamp >= t:
                self.future_A_gamma_time = 1

        dy = xp[j] - curve_v2_math.newton_y(A_gamma[0], A_gamma[1], xp, self.D, j)
        # Not defining new "y" here to have less variables // make subsequent calls cheaper
        xp[j] -= dy
        dy -= 1

        if j > 0:
            dy = dy * PRECISION // price_scale
        dy //= prec_j

        dy -= self._fee(xp) * dy // 10**10
        assert dy >= min_dy, "Slippage"
        y -= dy

        self.balances[j] = y

        y *= prec_j
        if j > 0:
            y = y * price_scale // PRECISION
        xp[j] = y

        # Calculate price
        if dx > 10**5 and dy > 10**5:
            _dx: int = dx * prec_i
            _dy: int = dy * prec_j
            if i == 0:
                p = _dx * 10**18 // _dy
            else:  # j == 0
                p = _dy * 10**18 // _dx

        self.tweak_price(A_gamma, xp, p, 0)

        return dy

    @call_by_value
    def add_liquidity(self, amounts: list, min_mint_amount: int) -> int:
        assert not self.is_killed  # dev: the pool is killed

        A_gamma: list = self._A_gamma()

        xp: list = shallow_array_copy(self.balances)
        amountsp: list = empty(int, N_COINS)
        xx: list = empty(int, N_COINS)
        d_token: int = 0
        d_token_fee: int = 0
        old_D: int = 0

        xp_old: list = shallow_array_copy(xp)

        for i in range(N_COINS):
            bal: int = xp[i] + amounts[i]
            xp[i] = bal
            self.balances[i] = bal
        xx = shallow_array_copy(xp)

        price_scale: int = self.price_scale * self.precisions[1]
        xp = [xp[0] * self.precisions[0], xp[1] * price_scale // PRECISION]
        xp_old = [xp_old[0] * self.precisions[0], xp_old[1] * price_scale // PRECISION]

        for i in range(N_COINS):
            if amounts[i] > 0:
                amountsp[i] = xp[i] - xp_old[i]
        assert amounts[0] > 0 or amounts[1] > 0  # dev: no coins to add

        t: int = self.future_A_gamma_time
        if t > 0:
            old_D = curve_v2_math.newton_D(A_gamma[0], A_gamma[1], xp_old)
            if self.block.timestamp >= t:
                self.future_A_gamma_time = 1
        else:
            old_D = self.D

        D: int = curve_v2_math.newton_D(A_gamma[0], A_gamma[1], xp)

        token_supply: int = self.total_supply
        if old_D > 0:
            d_token = token_supply * D // old_D - token_supply
        else:
            d_token = self.get_xcp(D)  # making initial virtual price equal to 1
        assert d_token > 0  # dev: nothing minted

        if old_D > 0:
            d_token_fee = self._calc_token_fee(amountsp, xp) * d_token // 10**10 + 1
            d_token -= d_token_fee
            token_supply += d_token
            self.total_supply += d_token

            # Calculate price
            # p_i * (dx_i - dtoken // token_supply * xx_i) = sum{k!=i}(p_k * (dtoken // token_supply * xx_k - dx_k))
            # Simplified for 2 coins
            p: int = 0
            if d_token > 10**5:
                if amounts[0] == 0 or amounts[1] == 0:
                    S: int = 0
                    precision: int = 0
                    ix: int = 0
                    if amounts[0] == 0:
                        S = xx[0] * self.precisions[0]
                        precision = self.precisions[1]
                        ix = 1
                    else:
                        S = xx[1] * self.precisions[1]
                        precision = self.precisions[0]
                    S = S * d_token // token_supply
                    p = S * PRECISION // (amounts[ix] * precision - d_token * xx[ix] * precision // token_supply)
                    if ix == 0:
                        p = (10**18)**2 // p

            self.tweak_price(A_gamma, xp, p, D)

        else:
            self.D = D
            self.virtual_price = 10**18
            self.xcp_profit = 10**18
            self.total_supply += d_token

        assert d_token >= min_mint_amount, "Slippage"

        return d_token
        
    def get_dy(self, i: int, j: int, dx: int) -> int:
        assert i != j  # dev: same input and output coin
        assert i < N_COINS  # dev: coin index out of range
        assert j < N_COINS  # dev: coin index out of range

        price_scale: int = self.price_scale * self.precisions[1]
        xp: list = shallow_array_copy(self.balances)

        A_gamma: list = self._A_gamma()
        D: int = self.D
        if self.future_A_gamma_time > 0:
            D = curve_v2_math.newton_D(A_gamma[0], A_gamma[1], self.xp())

        xp[i] += dx
        xp = [xp[0] * self.precisions[0], xp[1] * price_scale // PRECISION]

        y: int = curve_v2_math.newton_y(A_gamma[0], A_gamma[1], xp, D, j)
        dy: int = xp[j] - y - 1
        xp[j] = y
        if j > 0:
            dy = dy * PRECISION // price_scale
        else:
            dy //= self.precisions[0]
        dy -= self._fee(xp) * dy // 10**10

        return dy

    def xp(self) -> list:
        return [self.balances[0] * self.precisions[0],
                curve_v2_math.unsafe_div(self.balances[1] * self.precisions[1] * self.price_scale, PRECISION)]

    def _calc_token_fee(self, amounts: list, xp: list) -> int:
        # fee = sum(amounts_i - avg(amounts)) * fee' // sum(amounts)
        fee: int = self._fee(xp) * N_COINS // (4 * (N_COINS-1))
        S: int = 0
        for _x in amounts:
            S += _x
        avg: int = S // N_COINS
        Sdiff: int = 0
        for _x in amounts:
            if _x > avg:
                Sdiff += _x - avg
            else:
                Sdiff += avg - _x
        return fee * Sdiff // S + NOISE_FEE

    ### ADDITIONS
    def get_balances(self):
        return shallow_array_copy(self.balances)

    def set_balances(self, balances):
        self.balances = shallow_array_copy(balances)

    def get_balance(self, k):
        return self.balances[k]

    def set_balance(self, k, v):
        self.balances[k] = v
