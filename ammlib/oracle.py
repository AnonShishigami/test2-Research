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

