import numpy as np


class BaseOracle:

    def __init__(self, ):
        self.times = []
        self.prices = []
        self.length = 0

    def update(self, time, price):
        self.times.append(time)
        self.prices.append(price)
        self.length += 1

    def reset(self):
        self.times = []
        self.prices = []
        self.length = 0

    def get(self, current_time):
        return np.nan

    def get_last_timestamped_prices(self, lookback_calls, route, lookback_step=1):
        if not self.length:
            return []
        return [
            {
                "price": self.prices[-i * lookback_step - 1] if route == "01" else 1 / self.prices[-i * lookback_step - 1],
                "ts": self.times[-i * lookback_step - 1]
            }
            for i in range(min(int((self.length - 1) / lookback_step), lookback_calls))
        ]


class PerfectOracle(BaseOracle):

    def __init__(self, ):
        super().__init__()

    def get(self, current_time):
        return self.prices[-1]


class LaggedOracle(BaseOracle):

    def __init__(self, lag):
        super().__init__()
        self.lag = lag

    def get(self, current_time):
        t = 0
        while (self.times[self.length-1-t] > current_time - self.lag) and (t < self.length-1):
            t += 1
        return self.prices[self.length-1-t]

    def get_last_timestamped_prices(self, lookback_calls, route, lookback_step=1):
        return []


class SparseOracle(PerfectOracle):

    def __init__(self, min_period):
        super().__init__()
        self.min_period = min_period

    def update(self, time, price):
        if (not self.times) or (time >= self.times[-1] + self.min_period):
            super().update(time, price)

