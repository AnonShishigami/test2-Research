# @version 0.3.1
# (c) Curve.Fi, 2021
# Math for crypto pools
#
# Unless otherwise agreed on, only contracts owned by Curve DAO or
# Swiss Stake GmbH are allowed to call this contract.

from sandbox.strategies.curve_v2.vyper_utils import call_by_value, shallow_array_copy
from sandbox.strategies.curve_v2.curve_v2_config import (
    N_COINS, A_MULTIPLIER, MIN_GAMMA, MAX_GAMMA, MIN_A, MAX_A, EXP_PRECISION
)


@call_by_value
def sort_function(A0: list) -> list:
    """
    Insertion sort from high to low
    """
    A: list = [v for v in A0]
    for i in range(1, N_COINS):
        x: int = A[i]
        cur: int = i
        for j in range(N_COINS):
            y: int = A[cur-1]
            if y > x:
                break
            A[cur] = y
            cur -= 1
            if cur == 0:
                break
        A[cur] = x
    return A


@call_by_value
def geometric_mean(unsorted_x: list, sort: bool) -> int:
    """
    (x[0] * x[1] * ...) ** (1/N)
    """
    x: list = shallow_array_copy(unsorted_x)
    if sort and x[0] < x[1]:
        x = [unsorted_x[1], unsorted_x[0]]
    D: int = x[0]
    diff: int = 0
    for i in range(255):
        D_prev: int = D
        # tmp: int = 10**18
        # for _x in x:
        #     tmp = tmp * _x // D
        # D = D * ((N_COINS - 1) * 10**18 + tmp) // (N_COINS * 10**18)
        # line below makes it for 2 coins
        D = (D + x[0] * x[1] // D) // N_COINS
        if D > D_prev:
            diff = D - D_prev
        else:
            diff = D_prev - D
        if diff <= 1 or diff * 10**18 < D:
            return D
    raise Exception("Did not converge")


@call_by_value
def newton_D(ANN: int, gamma: int, x_unsorted: list) -> int:
    """
    Finding the invariant using Newton method.
    ANN is higher by the factor A_MULTIPLIER
    ANN is already A * N**N

    Currently uses 60k gas
    """
    # Safety checks
    assert ANN > MIN_A - 1 and ANN < MAX_A + 1  # dev: unsafe values A
    assert gamma > MIN_GAMMA - 1 and gamma < MAX_GAMMA + 1  # dev: unsafe values gamma

    # Initial value of invariant D is that for constant-product invariant
    x: list = shallow_array_copy(x_unsorted)
    if x[0] < x[1]:
        x = [x_unsorted[1], x_unsorted[0]]

    assert x[0] > 10**9 - 1 and x[0] < 10**15 * 10**18 + 1  # dev: unsafe values x[0]
    assert x[1] * 10**18 // x[0] > 10**14-1  # dev: unsafe values x[i] (input)

    D: int = N_COINS * geometric_mean(x, False)
    S: int = x[0] + x[1]

    for i in range(255):
        D_prev: int = D

        # K0: int = 10**18
        # for _x in x:
        #     K0 = K0 * _x * N_COINS // D
        # collapsed for 2 coins
        K0: int = (10**18 * N_COINS**2) * x[0] // D * x[1] // D

        _g1k0: int = gamma + 10**18
        if _g1k0 > K0:
            _g1k0 = _g1k0 - K0 + 1
        else:
            _g1k0 = K0 - _g1k0 + 1

        # D // (A * N**N) * _g1k0**2 // gamma**2
        mul1: int = 10**18 * D // gamma * _g1k0 // gamma * _g1k0 * A_MULTIPLIER // ANN

        # 2*N*K0 // _g1k0
        mul2: int = (2 * 10**18) * N_COINS * K0 // _g1k0

        neg_fprime: int = (S + S * mul2 // 10**18) + mul1 * N_COINS // K0 - mul2 * D // 10**18

        # D -= f // fprime
        D_plus: int = D * (neg_fprime + S) // neg_fprime
        D_minus: int = D*D // neg_fprime
        if 10**18 > K0:
            D_minus += D * (mul1 // neg_fprime) // 10**18 * (10**18 - K0) // K0
        else:
            D_minus -= D * (mul1 // neg_fprime) // 10**18 * (K0 - 10**18) // K0

        if D_plus > D_minus:
            D = D_plus - D_minus
        else:
            D = (D_minus - D_plus) // 2

        diff: int = 0
        if D > D_prev:
            diff = D - D_prev
        else:
            diff = D_prev - D
        if diff * 10**14 < max(10**16, D):  # Could reduce precision for gas efficiency here
            # Test that we are safe with the next newton_y
            for _x in x:
                frac: int = _x * 10**18 // D
                assert (frac > 10**16 - 1) and (frac < 10**20 + 1)  # dev: unsafe values x[i]
            return D

    raise Exception("Did not converge")


@call_by_value
def newton_y(ANN: int, gamma: int, x: list, D: int, i: int) -> int:
    """
    Calculating x[i] given other balances x[0..N_COINS-1] and invariant D
    ANN = A * N**N
    """
    # Safety checks
    assert ANN > MIN_A - 1 and ANN < MAX_A + 1  # dev: unsafe values A
    assert gamma > MIN_GAMMA - 1 and gamma < MAX_GAMMA + 1  # dev: unsafe values gamma
    assert D > 10**17 - 1 and D < 10**15 * 10**18 + 1 # dev: unsafe values D

    x_j: int = x[1 - i]
    y: int = D**2 // (x_j * N_COINS**2)
    K0_i: int = (10**18 * N_COINS) * x_j // D
    # S_i = x_j

    # frac = x_j * 1e18 // D => frac = K0_i // N_COINS
    assert (K0_i > 10**16*N_COINS - 1) and (K0_i < 10**20*N_COINS + 1)  # dev: unsafe values x[i]

    # x_sorted: list = x
    # x_sorted[i] = 0
    # x_sorted = sort(x_sorted)  # From high to low
    # x[not i] instead of x_sorted since x_soted has only 1 element

    convergence_limit: int = max(max(x_j // 10**14, D // 10**14), 100)

    for j in range(255):
        y_prev: int = y

        K0: int = K0_i * y * N_COINS // D
        S: int = x_j + y

        _g1k0: int = gamma + 10**18
        if _g1k0 > K0:
            _g1k0 = _g1k0 - K0 + 1
        else:
            _g1k0 = K0 - _g1k0 + 1

        # D // (A * N**N) * _g1k0**2 // gamma**2
        mul1: int = 10**18 * D // gamma * _g1k0 // gamma * _g1k0 * A_MULTIPLIER // ANN

        # 2*K0 // _g1k0
        mul2: int = 10**18 + (2 * 10**18) * K0 // _g1k0

        yfprime: int = 10**18 * y + S * mul2 + mul1
        _dyfprime: int = D * mul2
        if yfprime < _dyfprime:
            y = y_prev // 2
            continue
        else:
            yfprime -= _dyfprime
        fprime: int = yfprime // y

        # y -= f // f_prime;  y = (y * fprime - f) // fprime
        # y = (yfprime + 10**18 * D - 10**18 * S) // fprime + mul1 // fprime * (10**18 - K0) // K0
        y_minus: int = mul1 // fprime
        y_plus: int = (yfprime + 10**18 * D) // fprime + y_minus * 10**18 // K0
        y_minus += 10**18 * S // fprime

        if y_plus < y_minus:
            y = y_prev // 2
        else:
            y = y_plus - y_minus

        diff: int = 0
        if y > y_prev:
            diff = y - y_prev
        else:
            diff = y_prev - y
        if diff < max(convergence_limit, y // 10**14):
            frac: int = y * 10**18 // D
            assert (frac > 10**16 - 1) and (frac < 10**20 + 1)  # dev: unsafe value for y
            return y

    raise Exception("Did not converge")


def halfpow(power: int) -> int:
    """
    1e18 * 0.5 ** (power/1e18)

    Inspired by: https://github.com/balancer-labs/balancer-core/blob/master/contracts/BNum.sol#L128
    """
    intpow: int = power // 10**18
    otherpow: int = power - intpow * 10**18
    if intpow > 59:
        return 0
    result: int = 10**18 // (2**intpow)
    if otherpow == 0:
        return result

    term: int = 10**18
    x: int = 5 * 10**17
    S: int = 10**18
    neg: bool = False

    for i in range(1, 256):
        K: int = i * 10**18
        c: int = K - 10**18
        if otherpow > c:
            c = otherpow - c
            neg = not neg
        else:
            c -= otherpow
        term = term * (c * x // 10**18) // K
        if neg:
            S -= term
        else:
            S += term
        if term < EXP_PRECISION:
            return result * S // 10**18

    raise Exception("Did not converge")
