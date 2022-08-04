# @version 0.3.1
# (c) Curve.Fi, 2021
# Math for crypto pools
#
# Unless otherwise agreed on, only contracts owned by Curve DAO or
# Swiss Stake GmbH are allowed to call this contract.

from .vyper_utils import call_by_value


N_COINS: int = 3  # <- change
A_MULTIPLIER: int = 10000

MIN_GAMMA: int = 10 ** 10
MAX_GAMMA: int = 5 * 10 ** 16

MIN_A: int = N_COINS ** N_COINS * A_MULTIPLIER // 100
MAX_A: int = N_COINS ** N_COINS * A_MULTIPLIER * 1000

@call_by_value
def sort_function(A0: list[int]) -> list[int]:
    """
    Insertion sort from high to low
    """
    A: list[int] = [v for v in A0]
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
def _geometric_mean(unsorted_x: list[int], sort: bool = True) -> int:
    """
    (x[0] * x[1] * ...) ** (1/N)
    """
    x: list[int] = [v for v in unsorted_x]
    if sort:
        x = sort_function(x)
    D: int = x[0]
    diff: int = 0
    for i in range(255):
        D_prev: int = D
        tmp: int = 10**18
        for _x in x:
            tmp = tmp * _x // D
        D = D * ((N_COINS - 1) * 10**18 + tmp) // (N_COINS * 10**18)
        if D > D_prev:
            diff = D - D_prev
        else:
            diff = D_prev - D
        if diff <= 1 or diff * 10**18 < D:
            return D
    raise ValueError("Did not converge")


@call_by_value
def geometric_mean(unsorted_x: list[int], sort: bool = True) -> int:
    return _geometric_mean(unsorted_x, sort)


@call_by_value
def reduction_coefficient(x: list[int], fee_gamma: int) -> int:
    """
    fee_gamma // (fee_gamma + (1 - K))
    where
    K = prod(x) // (sum(x) // N)**N
    (all normalized to 1e18)
    """
    K: int = 10**18
    S: int = 0
    for x_i in x:
        S += x_i
    # Could be good to pre-sort x, but it is used only for dynamic fee,
    # so that is not so important
    for x_i in x:
        K = K * N_COINS * x_i // S
    if fee_gamma > 0:
        K = fee_gamma * 10**18 // (fee_gamma + 10**18 - K)
    return K


@call_by_value
def newton_D(ANN: int, gamma: int, x_unsorted: list[int]) -> int:
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
    x: list[int] = sort_function(x_unsorted)

    assert x[0] > 10**9 - 1 and x[0] < 10**15 * 10**18 + 1  # dev: unsafe values x[0]
    for i in range(1, N_COINS):
        frac: int = x[i] * 10**18 // x[0]
        assert frac > 10**11-1  # dev: unsafe values x[i]

    D: int = N_COINS * _geometric_mean(x, False)
    S: int = 0
    for x_i in x:
        S += x_i

    for i in range(255):
        D_prev: int = D

        K0: int = 10**18
        for _x in x:
            K0 = K0 * _x * N_COINS // D

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

    raise ValueError("Did not converge")

@call_by_value
def newton_y(ANN: int, gamma: int, x: list[int], D: int, i: int) -> int:
    """
    Calculating x[i] given other balances x[0..N_COINS-1] and invariant D
    ANN = A * N**N
    """
    # Safety checks
    assert ANN > MIN_A - 1 and ANN < MAX_A + 1  # dev: unsafe values A
    assert gamma > MIN_GAMMA - 1 and gamma < MAX_GAMMA + 1  # dev: unsafe values gamma
    assert D > 10**17 - 1 and D < 10**15 * 10**18 + 1 # dev: unsafe values D
    for k in range(N_COINS):
        if k != i:
            frac: int = x[k] * 10**18 // D
            assert (frac > 10**16 - 1) and (frac < 10**20 + 1)  # dev: unsafe values x[i]

    y: int = D // N_COINS
    K0_i: int = 10**18
    S_i: int = 0

    x_sorted: list[int] = [v for v in x]
    x_sorted[i] = 0
    x_sorted = sort_function(x_sorted)  # From high to low

    convergence_limit: int = max(max(x_sorted[0] // 10**14, D // 10**14), 100)
    for j in range(2, N_COINS+1):
        _x: int = x_sorted[N_COINS-j]
        y = y * D // (_x * N_COINS)  # Small _x first
        S_i += _x
    for j in range(N_COINS-1):
        K0_i = K0_i * x_sorted[j] * N_COINS // D  # Large _x first

    for j in range(255):
        y_prev: int = y

        K0: int = K0_i * y * N_COINS // D
        S: int = S_i + y

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

    raise ValueError("Did not converge")


def halfpow(power: int, precision: int) -> int:
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
        if term < precision:
            return result * S // 10**18

    raise ValueError("Did not converge")


def sqrt_int(x: int) -> int:
    """
    Originating from: https://github.com/vyperlang/vyper/issues/1266
    """

    if x == 0:
        return 0

    z: int = (x + 10**18) // 2
    y: int = x

    for i in range(256):
        if z == y:
            return y
        y = z
        z = (x * 10**18 // z + z) // 2

    raise ValueError("Did not converge")


if __name__ == "__main__":

    factor = 1000

    GT = 13909500825447947762 / factor

    PRECISION = 10**18

    precisions = [
        1,
        1,
        1,
    ]

    D = 36563412680673942970083170
    i = 1
    j = 2

    dx = 1*10**18 / factor

    A_gamma = [
        5400000,
        20000000000000,
    ]

    price_scale = [
        22239416879670065382803,
        1593854743135437389490,
    ]

    xp = [
        12441069707852459963468772,
        54255933122*10**10,
        7565098356774253252646,
    ]
    xp[i] += dx

    for k in range(1, N_COINS):
        xp[k] = xp[k] * price_scale[k - 1] * precisions[k] // PRECISION

    print("xp:", xp)
    dy = xp[j] - newton_y(A_gamma[0], A_gamma[1], xp, D, j)
    print("raw dy:", dy)

    dy = dy * PRECISION // price_scale[j - 1]
    print("dy:", dy)

    print(f"error: {(GT / dy - 1) * 100}%")
