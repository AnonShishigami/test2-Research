import unittest

import numpy as np

from tests.strategy_test_template import StrategyTestTemplate
from tests.strategy_test_oracle import StrategyTestOracle
from sandbox.strategies.curve_v2.strategy import CurveV2


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
        mid_fee = state["mid_fee"]
        out_fee = state["out_fee"]
        allowed_extra_profit = state["allowed_extra_profit"]
        fee_gamma = state["fee_gamma"]
        admin_fee = state["admin_fee"]
        ma_half_time = state["ma_half_time"]
        
        # hardcoded
        not_adjusted = True
        precisions = [10 ** 12, 1]
                
        strategy = CurveV2(
            name="curve_v2", 
            initial_inventories=balances,  # will be overwritten right below
            A=A, gamma=gamma,

            mid_fee=mid_fee, out_fee=out_fee, allowed_extra_profit=allowed_extra_profit,
            fee_gamma=fee_gamma, adjustment_step=adjustment_step,
            admin_fee=admin_fee, ma_half_time=ma_half_time,

            initial_cash=0, market=None, oracle=StrategyTestOracle(), support_arb=False, 
            initial_prices=[1, 1],  # hackish way of faking buy quote (to comply with Curve's interface) with sell quote (to comply with this tool's interface)
            precisions=precisions,
            dt_sim=1,
            input_precision_factor=1
        )

        strategy.smart_contract.set_balances(balances)
        strategy.smart_contract.D = D
        strategy.smart_contract.future_A_gamma_time = future_A_gamma_time
        
        strategy.smart_contract.price_scale = price_scale
        strategy.smart_contract.last_prices = last_prices
        strategy.smart_contract.last_prices_timestamp = last_prices_timestamp
        strategy.smart_contract.price_oracle = price_oracle
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
                    "D": 15390672454698502979973257,
                    "A": 200000000,
                    "gamma": 200000000000000,
                    "price_scale": 987655216778603916,
                    "balances": [8261874417624, 7220619435590432462231048],
                    "last_prices": 1010763417490556501,
                    "last_prices_timestamp": 1662967912,
                    "price_oracle": 1016687427501964241,
                    "virtual_price": 1005978051045569641,
                    "future_A_gamma_time": 0,
                    "total_supply": 7697264559353135143467051,
                    "xcp_profit": 1011896327615718437,
                    "xcp_profit_a": 1011896327615718437,
                    "adjustment_step": 1000000000000000,
                    "mid_fee": 5000000,
                    "out_fee": 45000000,
                    "allowed_extra_profit": 10000000000,
                    "fee_gamma": 5000000000000000,
                    "admin_fee": 5000000000,
                    "ma_half_time": 2000,
                },
                "events": [
                    {
                        "block": 15519742,
                        "type": "TokenExchange",
                        "sold_id": 1,
                        "tokens_sold": 17610763686547556188457,
                        "bought_id": 0,
                        "tokens_bought": 17758166916,
                        "timestamp": 1662968875,
                        "state": {
                            "D": 15397947078550033149048456,
                            "A": 200000000,
                            "gamma": 200000000000000,
                            "price_scale": 988642871995382519,
                            "balances": [8244116250708, 7238230199276980018419505],
                            "last_prices": 1008370064585276425,
                            "last_prices_timestamp": 1662968875,
                            "price_oracle": 1015006386351993726,
                            "virtual_price": 1005950691450940233,
                            "future_A_gamma_time": 0,
                            "total_supply": 7697264559353135143467051,
                            "xcp_profit": 1011899238035454493,
                            "xcp_profit_a": 1011896327615718437,
                            "adjustment_step": 1000000000000000,
                            "mid_fee": 5000000,
                            "out_fee": 45000000,
                            "allowed_extra_profit": 10000000000,
                            "fee_gamma": 5000000000000000,
                            "admin_fee": 5000000000,
                            "ma_half_time": 2000,
                        },
                    },
                ]
            }
        ]

    def _get_exchange_args(self, event):
        if event["type"] == "TokenExchange":
            return (
                event["tokens_sold"] / event["tokens_bought"],
                event["bought_id"],
                event["tokens_sold"],  # hackish way of faking buy quote (to comply with Curve's interface) with sell quote (to comply with this tool's interface)
                event["timestamp"],
            )
        return None
    
    def _generate_state_getter(self):
        return lambda: dict((
            ("balances", self.strategy.smart_contract.balances),
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
            ("price_scale", self.strategy.smart_contract.price_scale),
            ("last_prices", self.strategy.smart_contract.last_prices),
            ("price_oracle", self.strategy.smart_contract.price_oracle),
            ("mid_fee", self.strategy.smart_contract.mid_fee),
            ("out_fee", self.strategy.smart_contract.out_fee),
            ("allowed_extra_profit", self.strategy.smart_contract.allowed_extra_profit),
            ("fee_gamma", self.strategy.smart_contract.fee_gamma),
            ("admin_fee", self.strategy.smart_contract.admin_fee),
            ("ma_half_time", self.strategy.smart_contract.ma_half_time),
        ))


if __name__ == "__main__":
    unittest.main()
