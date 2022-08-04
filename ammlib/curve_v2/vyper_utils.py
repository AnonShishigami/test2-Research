import copy


MUTABLE_TYPES_OF_INTEREST = [list, dict, set]


def call_by_value(func):
    def inner(*args, **kwargs):
        return func(
            *[copy.deepcopy(v) if type(v) in MUTABLE_TYPES_OF_INTEREST else v for v in args],
            **dict((k, copy.deepcopy(v) if type(v) in MUTABLE_TYPES_OF_INTEREST else v) for k, v in kwargs.items()),
        )
    return inner


def shallow_array_copy(l: list):
    return [v for v in l]


def shift(x, y):
    int_x = int(x)
    assert int_x == x
    if y >= 0:
        return int_x << y
    else:
        return int_x >> -y


def bitwise_or(x, y):
    int_x = int(x)
    assert int_x == x
    return int_x | y


def bitwise_and(x, y):
    int_x = int(x)
    assert int_x == x
    return int_x & y


def empty(t, size):
    return [0 for _ in range(size)]
