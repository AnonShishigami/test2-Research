# @version 0.3.1
# (c) Curve.Fi, 2021
# Pool for USDT/BTC/ETH or similar


from copy import deepcopy
import numpy as np

from ammlib.curve_v2.vyper_utils import shift, bitwise_or, bitwise_and, empty, call_by_value, shallow_array_copy
import ammlib.curve_v2.curve_v2_math as curve_v2_math


N_COINS: int = 3  # <- change
PRECISION: int = 10 ** 18  # The precision to convert to
A_MULTIPLIER: int = 10000

# These addresses are replaced by the deployer

KILL_DEADLINE_DT: int = 2 * 30 * 86400
ADMIN_ACTIONS_DELAY: int = 3 * 86400
MIN_RAMP_TIME: int = 86400

MAX_ADMIN_FEE: int = 10 * 10 ** 9
MIN_FEE: int = 5 * 10 ** 5  # 0.5 bps
MAX_FEE: int = 10 * 10 ** 9
MIN_A: int = N_COINS**N_COINS * A_MULTIPLIER // 100
MAX_A: int = 1000 * A_MULTIPLIER * N_COINS**N_COINS
MAX_A_CHANGE: int = 10
MIN_GAMMA: int = 10**10
MAX_GAMMA: int = 5 * 10**16
NOISE_FEE: int = 10**5  # 0.1 bps

PRICE_SIZE: int = 256 // (N_COINS-1)
PRICE_MASK: int = 2**PRICE_SIZE - 1

# This must be changed for different N_COINS
# For example:
# N_COINS = 3 -> 1  (10**18 -> 10**18)
# N_COINS = 4 -> 10**8  (10**18 -> 10**10)
# PRICE_PRECISION_MUL: int = 1
PRECISIONS = [
    1,#0
    1,#1
    1,#2
]

INF_COINS: int = 15


class Block:

    timestamp = 1630091674


block = Block()


class Swap:

    @call_by_value
    def __init__(
        self,
        owner: str,
        admin_fee_receiver: str,
        A: int,
        gamma: int,
        mid_fee: int,
        out_fee: int,
        allowed_extra_profit: int,
        fee_gamma: int,
        adjustment_step: int,
        admin_fee: int,
        ma_half_time: int,
        initial_prices: list[int],  # [N_COINS - 1]
        initial_balances: list[int],
    ):

        ### added by me
        self._balances = shallow_array_copy(initial_balances)
        ###

        self.owner = owner

        # Pack A and gamma:
        # shifted A + gamma
        A_gamma: int = shift(A, 128)
        A_gamma = bitwise_or(A_gamma, gamma)
        self.initial_A_gamma = A_gamma
        self.future_A_gamma = A_gamma

        self.mid_fee = mid_fee
        self.out_fee = out_fee
        self.allowed_extra_profit = allowed_extra_profit
        self.fee_gamma = fee_gamma
        self.adjustment_step = adjustment_step
        self.admin_fee = admin_fee

        # Packing prices
        packed_prices: int = 0
        for k in range(N_COINS-1):
            packed_prices = shift(packed_prices, PRICE_SIZE)
            p: int = initial_prices[N_COINS-2 - k]  # // PRICE_PRECISION_MUL
            assert p < PRICE_MASK
            packed_prices = bitwise_or(p, packed_prices)

        self.price_scale_packed = packed_prices
        self.price_oracle_packed = packed_prices
        self.last_prices_packed = packed_prices
        self.last_prices_timestamp = block.timestamp
        self.ma_half_time = ma_half_time

        self.xcp_profit_a = 10**18

        self.kill_deadline = block.timestamp + KILL_DEADLINE_DT

        self.admin_fee_receiver = admin_fee_receiver

        # hardcoded
        self.initial_A_gamma = 1837524781373067702702222880131558341862400000
        self.future_A_gamma = 1837524781373067702702222880131568341862400000
        self.initial_A_gamma_time = 1642374454
        self.future_A_gamma_time = 0
        self.D = len(initial_balances) * np.prod([b * p // PRECISION for b, p in zip (initial_balances, [PRECISION] + initial_prices)]) ** (1 / len(initial_balances))
        self.initial_supply = 100 * PRECISION
        self.xcp_profit = 0
        self.virtual_price = 0
        self.not_adjusted = False
        self.is_killed = False

    def set_future_A_gamma_time(self, future_A_gamma_time):
        self.future_A_gamma_time = future_A_gamma_time

    def set_D(self, D):
        self.D = D

    def get_balances(self):
        return shallow_array_copy(self._balances)

    def set_balances(self, balances):
        self._balances = shallow_array_copy(balances)

    def get_balance(self, k):
        return self._balances[k]

    def set_balance(self, k, v):
        self._balances[k] = v

    @staticmethod
    def _packed_view(k: int, p: int) -> int:
        assert k < N_COINS-1
        r = bitwise_and(
            shift(p, -PRICE_SIZE * k),
            PRICE_MASK
        )
        return bitwise_and(
            shift(p, -PRICE_SIZE * k),
            PRICE_MASK
        )  # * PRICE_PRECISION_MUL

    def price_oracle(self, k: int) -> int:
        return self._packed_view(k, self.price_oracle_packed)

    def price_scale(self, k: int) -> int:
        return self._packed_view(k, self.price_scale_packed)

    def last_prices(self, k: int) -> int:
        return self._packed_view(k, self.last_prices_packed)

    def _A_gamma(self) -> list[int]:
        t1: int = self.future_A_gamma_time

        A_gamma_1: int = self.future_A_gamma
        gamma1: int = bitwise_and(A_gamma_1, 2**128-1)
        A1: int = shift(A_gamma_1, -128)

        if block.timestamp < t1:
            # handle ramping up and down of A
            A_gamma_0: int = self.initial_A_gamma
            t0: int = self.initial_A_gamma_time

            # Less readable but more compact way of writing and converting to int
            # gamma0: int = bitwise_and(A_gamma_0, 2**128-1)
            # A0: int = shift(A_gamma_0, -128)
            # A1 = A0 + (A1 - A0) * (block.timestamp - t0) // (t1 - t0)
            # gamma1 = gamma0 + (gamma1 - gamma0) * (block.timestamp - t0) // (t1 - t0)

            t1 -= t0
            t0 = block.timestamp - t0
            t2: int = t1 - t0

            A1 = (shift(A_gamma_0, -128) * t2 + A1 * t0) // t1
            gamma1 = (bitwise_and(A_gamma_0, 2**128-1) * t2 + gamma1 * t0) // t1

        return [A1, gamma1]

    def A(self) -> int:
        return self._A_gamma()[0]

    def gamma(self) -> int:
        return self._A_gamma()[1]

    def _fee(self, xp: list[int]) -> int:
        f: int = curve_v2_math.reduction_coefficient(xp, self.fee_gamma)
        return (self.mid_fee * f + self.out_fee * (10**18 - f)) // 10**18

    def fee_calc(self, xp: list[int]) -> int:
        return self._fee(xp)

    def get_xcp(self, D: int) -> int:
        x: list[int] = empty(int, N_COINS)
        x[0] = D // N_COINS
        packed_prices: int = self.price_scale_packed
        # No precisions here because we don't switch to "real" units

        for i in range(1, N_COINS):
            x[i] = D * 10**18 // (N_COINS * bitwise_and(packed_prices, PRICE_MASK))  # ... * PRICE_PRECISION_MUL)
            packed_prices = shift(packed_prices, -PRICE_SIZE)

        return curve_v2_math.geometric_mean(x)

    @call_by_value
    def tweak_price(self, A_gamma: list[int], _xp: list[int], i: int, p_i: int, new_D: int):
        price_oracle: list[int] = empty(int, N_COINS-1)
        last_prices: list[int] = empty(int, N_COINS-1)
        price_scale: list[int] = empty(int, N_COINS-1)
        xp: list[int] = empty(int, N_COINS)
        p_new: list[int] = empty(int, N_COINS-1)

        # Update MA if needed
        packed_prices: int = self.price_oracle_packed
        for k in range(N_COINS-1):
            price_oracle[k] = bitwise_and(packed_prices, PRICE_MASK)  # * PRICE_PRECISION_MUL
            packed_prices = shift(packed_prices, -PRICE_SIZE)

        last_prices_timestamp: int = self.last_prices_timestamp
        packed_prices = self.last_prices_packed
        for k in range(N_COINS-1):
            last_prices[k] = bitwise_and(packed_prices, PRICE_MASK)   # * PRICE_PRECISION_MUL
            packed_prices = shift(packed_prices, -PRICE_SIZE)

        if last_prices_timestamp < block.timestamp:
            # MA update required
            ma_half_time: int = self.ma_half_time
            alpha: int = curve_v2_math.halfpow((block.timestamp - last_prices_timestamp) * 10**18 // ma_half_time, 10**10)
            packed_prices = 0
            for k in range(N_COINS-1):
                price_oracle[k] = (last_prices[k] * (10**18 - alpha) + price_oracle[k] * alpha) // 10**18
            for k in range(N_COINS-1):
                packed_prices = shift(packed_prices, PRICE_SIZE)
                p: int = price_oracle[N_COINS-2 - k]  # // PRICE_PRECISION_MUL
                assert p < PRICE_MASK
                packed_prices = bitwise_or(p, packed_prices)
            self.price_oracle_packed = packed_prices
            self.last_prices_timestamp = block.timestamp

        D_unadjusted: int = new_D  # Withdrawal methods know new D already
        if new_D == 0:
            # We will need this a few times (35k gas)
            D_unadjusted = curve_v2_math.newton_D(A_gamma[0], A_gamma[1], _xp)
        packed_prices = self.price_scale_packed
        for k in range(N_COINS-1):
            price_scale[k] = bitwise_and(packed_prices, PRICE_MASK)  # * PRICE_PRECISION_MUL
            packed_prices = shift(packed_prices, -PRICE_SIZE)

        if p_i > 0:
            # Save the last price
            if i > 0:
                last_prices[i-1] = p_i
            else:
                # If 0th price changed - change all prices instead
                for k in range(N_COINS-1):
                    last_prices[k] = last_prices[k] * 10**18 // p_i
        else:
            # calculate real prices
            # it would cost 70k gas for a 3-token pool. Sad. How do we do better?
            __xp: list[int] = shallow_array_copy(_xp)
            dx_price: int = __xp[0] // 10**6
            __xp[0] += dx_price
            for k in range(N_COINS-1):
                last_prices[k] = price_scale[k] * dx_price // (_xp[k+1] - curve_v2_math.newton_y(A_gamma[0], A_gamma[1], __xp, D_unadjusted, k+1))

        packed_prices = 0
        for k in range(N_COINS-1):
            packed_prices = shift(packed_prices, PRICE_SIZE)
            p: int = last_prices[N_COINS-2 - k]  # // PRICE_PRECISION_MUL
            assert p < PRICE_MASK
            packed_prices = bitwise_or(p, packed_prices)
        self.last_prices_packed = packed_prices

        # total_supply: int = CurveToken(token).totalSupply(self)
        total_supply = self.initial_supply
        old_xcp_profit: int = self.xcp_profit
        old_virtual_price: int = self.virtual_price

        # Update profit numbers without price adjustment first
        xp[0] = D_unadjusted // N_COINS
        for k in range(N_COINS-1):
            xp[k+1] = D_unadjusted * 10**18 // (N_COINS * price_scale[k])
        xcp_profit: int = 10**18
        virtual_price: int = 10**18
        if old_virtual_price > 0:
            xcp: int = curve_v2_math.geometric_mean(xp)
            virtual_price = 10**18 * xcp // total_supply
            xcp_profit = old_xcp_profit * virtual_price // old_virtual_price

            t: int = self.future_A_gamma_time
            if virtual_price < old_virtual_price and t == 0:
                raise ValueError("Loss")
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
            norm: int = 0

            for k in range(N_COINS-1):
                ratio: int = price_oracle[k] * 10**18 // price_scale[k]
                if ratio > 10**18:
                    ratio -= 10**18
                else:
                    ratio = 10**18 - ratio
                norm += ratio**2

            if norm > adjustment_step ** 2 and old_virtual_price > 0:
                norm = curve_v2_math.sqrt_int(norm // 10**18)  # Need to convert to 1e18 units!

                for k in range(N_COINS-1):
                    p_new[k] = (price_scale[k] * (norm - adjustment_step) + adjustment_step * price_oracle[k]) // norm

                # Calculate balances*prices
                xp = shallow_array_copy(_xp)
                for k in range(N_COINS-1):
                    xp[k+1] = _xp[k+1] * p_new[k] // price_scale[k]

                # Calculate "extended constant product" invariant xCP and virtual price
                D: int = curve_v2_math.newton_D(A_gamma[0], A_gamma[1], xp)
                xp[0] = D // N_COINS
                for k in range(N_COINS-1):
                    xp[k+1] = D * 10**18 // (N_COINS * p_new[k])
                # We reuse old_virtual_price here but it's not old anymore
                old_virtual_price = 10**18 * curve_v2_math.geometric_mean(xp) // total_supply

                # Proceed if we've got enough profit
                # if (old_virtual_price > 10**18) and (2 * (old_virtual_price - 10**18) > xcp_profit - 10**18):
                if (old_virtual_price > 10**18) and (2 * old_virtual_price - 10**18 > xcp_profit):
                    packed_prices = 0
                    for k in range(N_COINS-1):
                        packed_prices = shift(packed_prices, PRICE_SIZE)
                        p: int = p_new[N_COINS-2 - k]  # // PRICE_PRECISION_MUL
                        assert p < PRICE_MASK
                        packed_prices = bitwise_or(p, packed_prices)
                    self.price_scale_packed = packed_prices
                    self.D = D
                    self.virtual_price = old_virtual_price

                    return

                else:
                    self.not_adjusted = False

        # If we are here, the price_scale adjustment did not happen
        # Still need to update the profit counter and D
        self.D = D_unadjusted
        self.virtual_price = virtual_price

    @call_by_value
    def exchange(self, i: int, j: int, dx: int, min_dy: int, use_eth: bool = False, apply: bool = False):
        assert not self.is_killed  # dev: the pool is killed
        assert i != j  # dev: coin index out of range
        assert i < N_COINS  # dev: coin index out of range
        assert j < N_COINS  # dev: coin index out of range
        assert dx > 0  # dev: do not exchange 0 coins

        balances = self.get_balances()

        A_gamma: list[int] = self._A_gamma()
        xp: list[int] = shallow_array_copy(balances)
        ix: int = j
        p: int = 0
        dy: int = 0

        if True:  # scope to reduce size of memory when making internal calls later

            y: int = xp[j]
            x0: int = xp[i]
            xp[i] = x0 + dx
            balances[i] = xp[i]

            price_scale: list[int] = empty(int, N_COINS-1)
            packed_prices: int = self.price_scale_packed
            for k in range(N_COINS-1):
                price_scale[k] = bitwise_and(packed_prices, PRICE_MASK)  # * PRICE_PRECISION_MUL
                packed_prices = shift(packed_prices, -PRICE_SIZE)

            precisions: list[int] = PRECISIONS
            xp[0] *= PRECISIONS[0]
            for k in range(1, N_COINS):
                xp[k] = xp[k] * price_scale[k-1] * precisions[k] // PRECISION
            
            prec_i: int = precisions[i]

            # In case ramp is happening
            D = self.D
            future_A_gamma_time = self.future_A_gamma_time
            if True:
                t: int = future_A_gamma_time
                if t > 0:
                    x0 *= prec_i
                    if i > 0:
                        x0 = x0 * price_scale[i-1] // PRECISION
                    x1: int = xp[i]  # Back up old value in xp
                    xp[i] = x0
                    D = curve_v2_math.newton_D(A_gamma[0], A_gamma[1], xp)
                    xp[i] = x1  # And restore
                    if block.timestamp >= t:
                        # self.future_A_gamma_time = 1  # TODO: REMOVE
                        future_A_gamma_time = 1

            prec_j: int = precisions[j]

            dy = xp[j] - curve_v2_math.newton_y(A_gamma[0], A_gamma[1], xp, D, j)
            # Not defining new "y" here to have less variables // make subsequent calls cheaper
            xp[j] -= dy
            dy -= 1

            if j > 0:
                dy = dy * PRECISION // price_scale[j-1]
            dy //=prec_j

            dy -= self._fee(xp) * dy // 10**10
            assert dy >= min_dy, "Slippage"
            y -= dy

            balances[j] = y
            # assert might be needed for some tokens - removed one to save bytespace

            y *= prec_j
            if j > 0:
                y = y * price_scale[j-1] // PRECISION
            xp[j] = y

            # Calculate price
            if dx > 10**5 and dy > 10**5:
                _dx: int = dx * prec_i
                _dy: int = dy * prec_j
                if i != 0 and j != 0:
                    p = bitwise_and(
                        shift(self.last_prices_packed, -PRICE_SIZE * i-1),
                        PRICE_MASK
                    ) * _dx // _dy  # * PRICE_PRECISION_MUL
                elif i == 0:
                    p = _dx * 10**18 // _dy
                else:  # j == 0
                    p = _dy * 10**18 // _dx
                    ix = i

        if apply:
            self.set_balances(balances)
            self.set_D(D)
            self.set_future_A_gamma_time(future_A_gamma_time)
            self.tweak_price(A_gamma, xp, ix, p, 0)

        return dy, A_gamma, xp, ix, p, balances, D, future_A_gamma_time

    def get_dy(self, i: int, j: int, dx: int) -> int:
        assert i != j and i < N_COINS and j < N_COINS, "coin index out of range"
        assert dx > 0, "do not exchange 0 coins"

        precisions: list[int] = PRECISIONS

        price_scale: list[int] = empty(int, N_COINS - 1)
        for k in range(N_COINS - 1):
            price_scale[k] = self.price_scale(k)
        xp: list[int] = empty(int, N_COINS)
        for k in range(N_COINS):
            xp[k] = self.get_balance(k)

        A: int = self.A()
        gamma: int = self.gamma()
        D: int = self.D
        if self.future_A_gamma_time > 0:
            _xp: list[int] = shallow_array_copy(xp)
            _xp[0] *= precisions[0]
            for k in range(N_COINS - 1):
                _xp[k + 1] = _xp[k + 1] * price_scale[k] * precisions[k + 1] // PRECISION
            D = curve_v2_math.newton_D(A, gamma, _xp)

        xp[i] += dx
        xp[0] *= precisions[0]
        for k in range(N_COINS - 1):
            xp[k + 1] = xp[k + 1] * price_scale[k] * precisions[k + 1] // PRECISION

        y: int = curve_v2_math.newton_y(A, gamma, xp, D, j)
        dy: int = xp[j] - y - 1
        xp[j] = y
        if j > 0:
            dy = dy * PRECISION // price_scale[j - 1]
        dy //= precisions[j]
        dy -= self.fee_calc(xp) * dy // 10 ** 10

        return dy

    @staticmethod
    def shallow_array_copy(arr: list):
        return [v for v in arr] 
