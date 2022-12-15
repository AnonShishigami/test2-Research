import numpy as np


class BaseOracle:

    def __init__(self, ):
        self.times = []
        self.prices = []
        self.length = 0
        self.current_time = 0

    def update(self, time, price):
        self.times.append(time)
        self.prices.append(price)
        self.length += 1
        self.current_time = time

    def reset(self):
        self.times = []
        self.prices = []
        self.length = 0

    def get(self):
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

    def __init__(self):
        super().__init__()

    def get(self):
        return self.prices[-1]


class LaggedOracle(BaseOracle):

    def __init__(self, lag):
        super().__init__()
        self.lag = lag

    def get(self):
        t = 0
        while (self.times[self.length-1-t] > self.current_time - self.lag) and (t < self.length-1):
            t += 1
        return self.prices[self.length-1-t]

    def get_last_timestamped_prices(self, lookback_calls, route, lookback_step=1):
        return []


class SparseOracle(PerfectOracle):

    def __init__(self, min_period, deviation_threshold, min_period_std=0):
        super().__init__()
        self.min_period = min_period
        self.deviation_threshold = deviation_threshold
        self.all_prices = []
        self.all_times = []

    def update(self, time, price):
        updated = False
        if (not self.times) or (time >= self.times[-1] + self.min_period):
            super().update(time, price)
            updated = True
        
        if self.deviation_threshold is not None:
            if not updated:
                if self.prices \
                    and len(self.all_prices) >= 1 \
                    and abs(self.all_prices[-1] / self.prices[-1] - 1) >= self.deviation_threshold \
                    and self.all_times[-1] >= self.times[-1]:
                        super().update(self.all_times[-1], self.all_prices[-1])
            self.all_prices.append(price)
            self.all_times.append(time)
