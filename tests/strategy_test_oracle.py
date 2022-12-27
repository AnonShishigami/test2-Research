class StrategyTestOracle:
    def __init__(self):
        self.current_time = 0
    
    def get(self):
        return 1
    
    def set_time(self, time):
        self.current_time = time
