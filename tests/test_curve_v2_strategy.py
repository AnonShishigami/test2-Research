import unittest

import numpy as np

from tests.strategy_test_template import StrategyTestTemplate
from tests.strategy_test_oracle import StrategyTestOracle
from sandbox.strategies.curve_v2.strategy import LiquidityProviderCurveV2


class TestCurveV2(StrategyTestTemplate, unittest.TestCase):

    def setUp(self):
        self.test_exchange_rtol = None  # equality between expected and actual values is required when set to None

    @staticmethod
    def _get_strategy(state):
        
        # corresponds to the state right before the first event was validated on chain
        D = state["D"]
        A = state["A"]
        gamma = state["gamma"]
        price_scale = state["price_scale"]
        balances = state["balances"]
        last_prices = state["last_prices"]
        last_prices_timestamp = state["last_prices_timestamp"]
        price_oracle = state["price_oracle"]
        virtual_price = state["virtual_price"]
        future_A_gamma_time = state["future_A_gamma_time"]
        total_supply = state["total_supply"]
        xcp_profit = state["xcp_profit"]
        xcp_profit_a = state["xcp_profit_a"]
        adjustment_step = state["adjustment_step"]
        
        # hardcoded
        not_adjusted = True
        precisions = [10 ** 12, 10 ** 10, 1]
                
        strategy = LiquidityProviderCurveV2(
            name="curve_v2", 
            initial_inventories=balances,  # will be overwritten right below
            A=A, gamma=gamma,
            initial_cash=0, market=None, oracle=StrategyTestOracle(), support_arb=False, 
            initial_prices=[1, 1, 1],  # hackish way of faking buy quote (to comply with Curve's interface) with sell quote (to comply with this tool's interface)
            precisions=precisions,
            dt_sim=1,
            input_precision_factor=1
        )

        strategy.smart_contract.set_balances(balances)
        strategy.smart_contract.D = D
        strategy.smart_contract.future_A_gamma_time = future_A_gamma_time
        
        strategy.smart_contract.price_scale_packed = strategy.smart_contract.pack_prices(price_scale)
        strategy.smart_contract.last_prices_packed = strategy.smart_contract.pack_prices(last_prices)
        strategy.smart_contract.last_prices_timestamp = last_prices_timestamp
        strategy.smart_contract.price_oracle_packed = strategy.smart_contract.pack_prices(price_oracle)
        strategy.smart_contract.xcp_profit = xcp_profit
        strategy.smart_contract.xcp_profit_a = xcp_profit_a
        strategy.smart_contract.virtual_price = virtual_price
        strategy.smart_contract.total_supply = total_supply
        strategy.smart_contract.adjustment_step = adjustment_step
        
        strategy.smart_contract.not_adjusted = not_adjusted

        return strategy
    
    @staticmethod
    def _get_scenarios():
        # each scenario's events should sorted in chronological order
        return [
            {
                # corresponds to the state right before the first event was validated on chain
                "init_state": {
                    "D": 261060990592972941853552934,
                    "A": 1707629,
                    "gamma": 11809167828997,
                    "price_scale": [21539932720675192810676, 1699864853713229400872],
                    "balances": [86372512675968, 406208864120, 51293610544089081651002],
                    "last_prices": [21467496094400000000000, 1692845955692876735836],
                    "last_prices_timestamp": 1661446304,
                    "price_oracle": [21525754826108556482334, 1699466274394784131114],
                    "virtual_price": 1021949543493453999,
                    "future_A_gamma_time": 0,
                    "total_supply": 256431689600498313210511,
                    "xcp_profit": 1043832045744687957,
                    "xcp_profit_a": 1043566090540808146,
                    "adjustment_step": 2000000000000000,
                },
                "events": [
                    {
                        "block": 15410251,
                        "type": "TokenExchange",
                        "sold_id": 2,
                        "tokens_sold": 118634129499999993856,
                        "bought_id": 1,
                        "tokens_bought": 936014609,
                        "timestamp": 1661446312,
                        "state": {
                            "D": 261061130400381923088444169,
                            "A": 1707629,
                            "gamma": 11809167828997,
                            "price_scale": [21539932720675192810676, 1699864853713229400872],
                            "balances": [86372512675968, 405272849511, 51412244673589081644858],
                            "last_prices": [21455787591368671736816, 1692845955692876735836],
                            "last_prices_timestamp": 1661446312,
                            "price_oracle": [21525218881804045034049, 1699405371556732530017],
                            "virtual_price": 1021950090783638797,
                            "future_A_gamma_time": 0,
                            "total_supply": 256431689600498313210511,
                            "xcp_profit": 1043832604753727934,
                            "xcp_profit_a": 1043566090540808146,
                            "adjustment_step": 2000000000000000,
                        },
                    },
                    {
                        "block": 15410271,
                        "type": "TokenExchange",
                        "sold_id": 1,
                        "tokens_sold": 512108000,
                        "bought_id": 2,
                        "tokens_bought": 64843530378856012723,
                        "timestamp": 1661446627,
                        "state": {
                            "D": 260815062395012578291509280,
                            "A": 1707629,
                            "gamma": 11809167828997,
                            "price_scale": [21507408828544127644367, 1697635438890754522343],
                            "balances": [86372512675968, 405784957511, 51347401143210225632135],
                            "last_prices": [21455787591368671736816, 1694491402248428196440],
                            "last_prices_timestamp": 1661446627,
                            "price_oracle": [21504039499539275573971, 1697404481538081561991],
                            "virtual_price": 1021948190737270899,
                            "future_A_gamma_time": 0,
                            "total_supply": 256431689600498313210511,
                            "xcp_profit": 1043832910715893643,
                            "xcp_profit_a": 1043566090540808146,
                            "adjustment_step": 2000000000000000,
                        },
                    },
                ]
            }
        ]

    def _get_exchange_args(self, event):
        if event["type"] == "TokenExchange":
            return (
                event["tokens_sold"] / event["tokens_bought"],
                1 if self.strategy.asset_1_index == event["bought_id"] else 0,
                event["tokens_sold"],  # hackish way of faking buy quote (to comply with Curve's interface) with sell quote (to comply with this tool's interface)
                event["timestamp"],
            )
        return None
    
    def _generate_state_getter(self):
        return lambda: dict((
            ("balances", self.strategy.smart_contract.get_balances()),
            ("A", self.strategy.smart_contract.A()),
            ("D", self.strategy.smart_contract.D),
            ("gamma", self.strategy.smart_contract.gamma()),
            ("future_A_gamma_time", self.strategy.smart_contract.future_A_gamma_time),
            ("last_prices_timestamp", self.strategy.smart_contract.last_prices_timestamp),
            ("xcp_profit", self.strategy.smart_contract.xcp_profit),
            ("xcp_profit_a", self.strategy.smart_contract.xcp_profit_a),
            ("virtual_price", self.strategy.smart_contract.virtual_price),
            ("total_supply", self.strategy.smart_contract.total_supply),
            ("adjustment_step", self.strategy.smart_contract.adjustment_step),
            ("price_scale", self.strategy.smart_contract.unpack_prices(self.strategy.smart_contract.price_scale_packed)),
            ("last_prices", self.strategy.smart_contract.unpack_prices(self.strategy.smart_contract.last_prices_packed)),
            ("price_oracle", self.strategy.smart_contract.unpack_prices(self.strategy.smart_contract.price_oracle_packed)),
        ))


if __name__ == "__main__":
    unittest.main()
