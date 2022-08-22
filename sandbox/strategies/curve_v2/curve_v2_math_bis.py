# @version 0.3.1
# (c) Curve.Fi, 2021
# Math for crypto pools
#
# Unless otherwise agreed on, only contracts owned by Curve DAO or
# Swiss Stake GmbH are allowed to call this contract.
import copy 

MUTABLE_TYPES_OF_INTEREST = [list, dict, set]


def call_by_value(func):
    def inner(*args, **kwargs):
        return func(
            *[copy.deepcopy(v) if type(v) in MUTABLE_TYPES_OF_INTEREST else v for v in args],
            **dict((k, copy.deepcopy(v) if type(v) in MUTABLE_TYPES_OF_INTEREST else v) for k, v in kwargs.items()),
        )
    return inner


N_COINS: int = 3  # <- change
A_MULTIPLIER: int = 10000

MIN_GAMMA: int = 10 ** 10
MAX_GAMMA: int = 5 * 10 ** 16

MIN_A: int = N_COINS ** N_COINS * A_MULTIPLIER // 100
MAX_A: int = N_COINS ** N_COINS * A_MULTIPLIER * 1000


@call_by_value
def geometric_mean(x: list[int]) -> int:
    N = len(x)
    x = sorted(x, reverse=True)  # Presort - good for convergence
    D = x[0]
    for i in range(255):
        D_prev = D
        tmp = 10 ** 18
        for _x in x:
            tmp = tmp * _x // D
        D = D * ((N - 1) * 10**18 + tmp) // (N * 10**18)
        diff = abs(D - D_prev)
        if diff <= 1 or diff * 10**18 < D:
            return D
    # print(x)
    raise ValueError("Did not converge")


@call_by_value
def reduction_coefficient(x: list[int], gamma: int) -> int:
    """
    fee_gamma // (fee_gamma + (1 - K))
    where
    K = prod(x) // (sum(x) // N)**N
    (all normalized to 1e18)
    """
    N = len(x)
    x_prod = 10**18
    K = 10**18
    S = sum(x)
    for x_i in x:
        x_prod = x_prod * x_i // 10**18
        K = K * N * x_i // S
    if gamma > 0:
        K = gamma * 10**18 // (gamma + 10**18 - K)
    return K


@call_by_value
def newton_D(A: int, gamma: int, x_unsorted: list[int]) -> int:
    """
    Finding the invariant using Newton method.
    ANN is higher by the factor A_MULTIPLIER
    ANN is already A * N**N
    Currently uses 60k gas
    """

    x = sorted(x_unsorted, reverse=True)

    D: int = N_COINS * geometric_mean(x)

    S = sum(x)
    N = len(x)

    for i in range(255):
        D_prev = D

        K0 = 10**18
        for _x in x:
            K0 = K0 * _x * N // D

        _g1k0 = abs(gamma + 10**18 - K0)

        # D / (A * N**N) * _g1k0**2 / gamma**2
        mul1 = 10**18 * D // gamma * _g1k0 // gamma * _g1k0 * A_MULTIPLIER // A

        # 2*N*K0 / _g1k0
        mul2 = (2 * 10**18) * N * K0 // _g1k0

        neg_fprime = (S + S * mul2 // 10**18) + mul1 * N // K0 - mul2 * D // 10**18
        assert neg_fprime > 0  # Python only: -f' > 0

        # D -= f / fprime
        D = (D * neg_fprime + D * S - D**2) // neg_fprime - D * (mul1 // neg_fprime) // 10**18 * (10**18 - K0) // K0

        if D < 0:
            D = -D // 2
        if abs(D - D_prev) <= max(100, D // 10**14):
            return D

    raise ValueError("Did not converge")


@call_by_value
def newton_y(A: int, gamma: int, x: list[int], D: int, i: int) -> int:
    """
    Calculating x[i] given other balances x[0..N_COINS-1] and invariant D
    ANN = A * N**N
    """
    N = len(x)

    y = D // N
    K0_i = 10**18
    S_i = 0
    x_sorted = sorted(_x for j, _x in enumerate(x) if j != i)
    convergence_limit = max(max(x_sorted) // 10**14, D // 10**14, 100)
    for _x in x_sorted:
        y = y * D // (_x * N)  # Small _x first
        S_i += _x
    for _x in x_sorted[::-1]:
        K0_i = K0_i * _x * N // D  # Large _x first

    for j in range(255):
        y_prev = y

        K0 = K0_i * y * N // D
        S = S_i + y

        _g1k0 = abs(gamma + 10**18 - K0)

        # D / (A * N**N) * _g1k0**2 / gamma**2
        mul1 = 10**18 * D // gamma * _g1k0 // gamma * _g1k0 * A_MULTIPLIER // A

        # 2*K0 / _g1k0
        mul2 = 10**18 + (2 * 10**18) * K0 // _g1k0

        yfprime = (10**18 * y + S * mul2 + mul1 - D * mul2)
        fprime = yfprime // y
        assert fprime > 0  # Python only: f' > 0

        # y -= f / f_prime;  y = (y * fprime - f) / fprime
        y = (yfprime + 10**18 * D - 10**18 * S) // fprime + mul1 // fprime * (10**18 - K0) // K0

        # if j > 100:  # Just logging when doesn't converge
        #     print(j, y, D, x)
        if y < 0 or fprime < 0:
            y = y_prev // 2
        if abs(y - y_prev) <= max(convergence_limit, y // 10**14):
            return y

    raise Exception("Did not converge")


def halfpow(power: int, precision: int) -> int:
    """
    1e18 * 0.5 ** (power/1e18)
    Inspired by: https://github.com/balancer-labs/balancer-core/blob/master/contracts/BNum.sol#L128
    """
    return 1e18 * 0.5 ** (power/1e18)


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
